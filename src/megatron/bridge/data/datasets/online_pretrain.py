# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Online pretrain dataset: on-the-fly tokenization, offline-style windowing.

Reads raw text from parquet or jsonl files and tokenizes at runtime, but
otherwise mirrors Megatron-Core's *offline* GPT MMap dataset
(``megatron.core.datasets.gpt_dataset.GPTDataset``) as closely as possible:

- Every document is tokenized with an EOD token appended (``append_eod``),
  exactly like ``Megatron-LM/tools/preprocess_data.py``'s ``Encoder.encode``
  (EOD is appended only when the document produced at least one token).
- Per split, documents are conceptually concatenated into one long token
  stream (document order reshuffled every epoch, mirroring
  ``_build_document_index``), and fixed ``seq_length + 1`` windows are
  sliced out of that stream as samples (mirroring
  ``_query_document_sample_shuffle_indices`` / ``sample_index``). A window
  may span multiple documents, or a partial document.
- ``reset_position_ids``, ``reset_attention_mask``, and ``eod_mask_loss``
  behave exactly as they do offline: document boundaries are found by
  scanning the sampled window for the EOD token *value*
  (see ``_build_ltor_mask_and_position_ids``, a direct port of
  ``gpt_dataset._get_ltor_masks_and_position_ids``) rather than via
  external cu_seqlens/THD packing metadata. This works because every
  document boundary is embedded directly in the token stream by
  ``append_eod``, and a per-sample window is small enough that an O(L^2)
  dense mask is affordable -- the same reasoning that lets the offline
  dataset skip cu_seqlens entirely.

Trade-off vs. the offline dataset: because tokenization happens at access
time (there is no pre-tokenized binary to mmap), a document that happens to
span multiple sample windows may be re-tokenized more than once if those
windows land far apart after per-epoch document shuffling. A bounded
per-worker LRU cache (see ``_tokenize_document``) absorbs the common case
where nearby windows share a document. In practice most corpora used for
online pretraining have documents much shorter than ``seq_length``, so a
window spans several small documents rather than the other way around, and
this cost is rare.
"""

from __future__ import annotations

import bisect
import glob
import gzip
import json
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


logger = logging.getLogger(__name__)

try:
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False

_MAX_CACHED_FILES = 32  # LRU bound on open parquet/jsonl handles per worker
_MAX_CACHED_DOCS = 512  # LRU bound on cached tokenized documents per worker

# Transient sentinel used to fill the final (partial) window at the end of a
# split's token stream. Never a valid token id, so it can't collide with a
# real EOD/vocab id; remapped to 0 (and its loss masked out) before a sample
# is returned. Mirrors megatron.core.datasets.megatron_dataset._PAD_TOKEN_ID.
_PAD_TOKEN_ID = -1


# ---------------------------------------------------------------------------
# File resolution & record counting
# ---------------------------------------------------------------------------


def is_parquet(file_name: str) -> bool:
    """Return True if *file_name* has a parquet extension (.parquet/.pq)."""
    return file_name.lower().endswith((".parquet", ".pq"))


def resolve_input_files(path: str) -> list[str]:
    """Resolve path to a sorted list of data files (recursive for directories)."""
    if os.path.isdir(path):
        files: list[str] = []
        for root, _, _ in os.walk(path):
            for ext in ("*.parquet", "*.pq", "*.jsonl", "*.jsonl.gz", "*.json"):
                files.extend(glob.glob(os.path.join(root, ext)))
        if not files:
            raise FileNotFoundError(f"No data files found in: {path}")
        return sorted(files)
    elif "*" in path or "?" in path:
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No files matching: {path}")
        return files
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")
        return [path]


def count_records(file_path: str) -> int:
    """Return the number of records in a file."""
    if is_parquet(file_path):
        if not _PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for parquet files")
        return pq.read_metadata(file_path).num_rows
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt") as f:
        return sum(1 for _ in f)


def count_records_batch(file_paths: list[str], num_workers: int = 1) -> list[int]:
    """Count records in each file, returning a list aligned with *file_paths*.

    Uses a thread pool since pyarrow metadata reads and file I/O release the
    GIL; on network filesystems (NFS / Lustre) with hundreds of files this is
    dramatically faster than a serial loop. ``num_workers`` is typically
    ``dataset.num_dataset_builder_threads``.
    """
    if len(file_paths) <= 1 or num_workers <= 1:
        return [count_records(f) for f in file_paths]

    counts = [0] * len(file_paths)
    with ThreadPoolExecutor(max_workers=min(num_workers, len(file_paths))) as pool:
        future_to_idx = {pool.submit(count_records, fp): i for i, fp in enumerate(file_paths)}
        for future in as_completed(future_to_idx):
            counts[future_to_idx[future]] = future.result()
    return counts


# ---------------------------------------------------------------------------
# Row-group-granularity parquet state (bounded LRU cache)
# ---------------------------------------------------------------------------


class _ParquetFileState:
    """Lazily-opened parquet reader with a single cached row group.

    Mirrors GPTSFTPackedParquetDataset's row-group cache in packed_parquet.py:
    memory footprint is bounded by one row group per open file, not the
    whole file.
    """

    __slots__ = ("pf", "row_group_offsets", "cached_rg_id", "cached_rg_table")

    def __init__(self, file_path: str) -> None:
        self.pf = pq.ParquetFile(file_path)
        meta = self.pf.metadata
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)
        self.row_group_offsets = offsets
        self.cached_rg_id: int | None = None
        self.cached_rg_table = None

    def read_rows(self, row_indices: list[int], text_key: str) -> list[str]:
        results: list[str | None] = [None] * len(row_indices)
        for pos, idx in enumerate(row_indices):
            rg_id = bisect.bisect_right(self.row_group_offsets, idx) - 1
            row_in_group = idx - self.row_group_offsets[rg_id]
            if self.cached_rg_id != rg_id:
                self.cached_rg_table = self.pf.read_row_group(rg_id, columns=[text_key])
                self.cached_rg_id = rg_id
            results[pos] = self.cached_rg_table.column(text_key)[row_in_group].as_py()
        return results  # type: ignore[return-value]


# Per-worker LRU caches: DataLoader workers are separate processes, so each
# gets its own. Bounded to _MAX_CACHED_FILES to avoid unbounded memory
# growth when reads are scattered across hundreds/thousands of files.
_parquet_states: "OrderedDict[str, _ParquetFileState]" = OrderedDict()
_jsonl_lines: "OrderedDict[str, list[str]]" = OrderedDict()


def _get_parquet_state(file_path: str) -> _ParquetFileState:
    if file_path in _parquet_states:
        _parquet_states.move_to_end(file_path)
        return _parquet_states[file_path]
    if len(_parquet_states) >= _MAX_CACHED_FILES:
        _parquet_states.popitem(last=False)  # evict least-recently-used
    state = _ParquetFileState(file_path)
    _parquet_states[file_path] = state
    return state


def _get_jsonl_lines(file_path: str) -> list[str]:
    if file_path in _jsonl_lines:
        _jsonl_lines.move_to_end(file_path)
        return _jsonl_lines[file_path]
    if len(_jsonl_lines) >= _MAX_CACHED_FILES:
        _jsonl_lines.popitem(last=False)
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt") as f:
        lines = f.readlines()
    _jsonl_lines[file_path] = lines
    return lines


def read_texts(file_path: str, row_indices: list[int], text_key: str = "text") -> list[str]:
    """Read text values at specific row indices from a file.

    Random-access hot path: uses row-group granularity for parquet and a
    bounded LRU cache for both formats, so memory stays bounded even when
    reads are scattered across many files.
    """
    if is_parquet(file_path):
        if not _PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for parquet files")
        return _get_parquet_state(file_path).read_rows(row_indices, text_key)
    lines = _get_jsonl_lines(file_path)
    return [json.loads(lines[i])[text_key] for i in row_indices]


def read_all_texts(file_path: str, text_key: str = "text") -> list[str]:
    """Read every text value from a file (one-shot full scan, e.g. for
    init-time length computation). Not cached — each file is typically
    visited exactly once during initialization.

    Uses pyarrow's native bulk ``to_pylist()`` conversion rather than a
    Python-level ``.as_py()`` loop, which is substantially faster.
    """
    if is_parquet(file_path):
        if not _PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for parquet files")
        table = pq.ParquetFile(file_path).read(columns=[text_key])
        return table.column(text_key).to_pylist()
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt") as f:
        return [json.loads(line)[text_key] for line in f]


# ---------------------------------------------------------------------------
# Tokenized-document cache
# ---------------------------------------------------------------------------

_token_cache: "OrderedDict[tuple, list[int]]" = OrderedDict()


def _tokenize_document(
    file_path: str,
    row_idx: int,
    tokenizer: MegatronTokenizer,
    text_key: str,
    append_eod: bool,
) -> list[int]:
    """Tokenize one document, appending EOD, with a bounded per-worker LRU cache.

    EOD is appended only if the document produced at least one token,
    exactly matching ``Megatron-LM/tools/preprocess_data.py``'s
    ``Encoder.encode`` (``if len(doc_ids) > 0 and args.append_eod: ...``) so
    that document-length accounting and boundary scanning stay consistent
    with how the offline binarizer builds its corpus.

    The cache avoids re-tokenizing the same document when consecutive
    (or nearby) sample windows draw from it, e.g. when one long document
    spans several ``seq_length`` windows.
    """
    key = (file_path, row_idx, text_key, append_eod)
    if key in _token_cache:
        _token_cache.move_to_end(key)
        return _token_cache[key]
    if len(_token_cache) >= _MAX_CACHED_DOCS:
        _token_cache.popitem(last=False)
    text = read_texts(file_path, [row_idx], text_key)[0]
    ids = list(tokenizer.tokenize(text))
    if append_eod and ids:
        ids.append(tokenizer.eod)
    _token_cache[key] = ids
    return ids


def _compute_file_lengths(
    file_path: str,
    tokenizer: MegatronTokenizer,
    text_key: str,
    append_eod: bool,
) -> list[int]:
    """Full (untruncated) per-document token counts for one file.

    Unlike a bin-packing scheme, documents are not truncated here: a very
    long document simply spans multiple sample windows, exactly like the
    offline GPT MMap dataset. EOD is counted only for non-empty documents
    (see ``_tokenize_document``).
    """
    texts = read_all_texts(file_path, text_key)
    lengths: list[int] = []
    for text in texts:
        n = sum(1 for _ in tokenizer.tokenize(text))
        if append_eod and n > 0:
            n += 1
        lengths.append(n)
    return lengths


# ---------------------------------------------------------------------------
# Distributed rank-0-build + sync helper
# ---------------------------------------------------------------------------
#
# Mirrors the rank-0-build -> barrier -> other-ranks pattern used by
# Megatron-Core's BlendedMegatronDatasetBuilder for offline GPT MMap
# datasets: only rank 0 performs the (expensive) build, then the result is
# either reloaded from a shared on-disk cache or broadcast in-memory.


def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank0_build_and_sync(build_fn, cache_path: str | None):
    """Build a picklable object on rank 0 and replicate it to all ranks.

    If *cache_path* is given, rank 0 persists the result there and other
    ranks reload from disk after a barrier (also serves as a cross-run
    cache). Otherwise the object is broadcast in-memory.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path, allow_pickle=True).tolist()

    rank = torch.distributed.get_rank() if _is_distributed() else 0
    obj = None
    if rank == 0:
        obj = build_fn()
        if cache_path:
            np.save(cache_path, np.array(obj, dtype=object), allow_pickle=True)

    if _is_distributed():
        if cache_path:
            torch.distributed.barrier()
            if rank > 0:
                obj = np.load(cache_path, allow_pickle=True).tolist()
        else:
            box = [obj]
            torch.distributed.broadcast_object_list(box, src=0)
            obj = box[0]
    return obj


def _scaled_builder_threads(num_dataset_builder_threads: int) -> int:
    """Scale up rank-0's thread count when it is the only rank building.

    Mirrors ``BlendedMegatronDatasetBuilder``: rank 0 builds alone while
    other ranks wait at the barrier, so it can safely use more threads
    (bounded to avoid overloading storage on a cache miss). A no-op when
    ``num_dataset_builder_threads <= 1`` (serial, matching the offline
    default) or when not running distributed.
    """
    num_workers = num_dataset_builder_threads
    if num_workers > 1 and _is_distributed() and torch.distributed.get_rank() == 0:
        num_workers *= min(2, max(1, torch.cuda.device_count()))
    return num_workers


# ---------------------------------------------------------------------------
# Offline-style mask / position-id construction
# ---------------------------------------------------------------------------


def _build_ltor_mask_and_position_ids(
    tokens: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Build attention mask, loss mask, and position ids for a token window.

    Direct port of Megatron-Core's
    ``megatron.core.datasets.gpt_dataset._get_ltor_masks_and_position_ids``
    (the function backing the offline GPT MMap dataset): document boundaries
    are found by scanning for occurrences of *eod_token* directly in the
    token values, not via external cu_seqlens/THD metadata. This only works
    because every document was terminated with an EOD token during
    tokenization (see ``append_eod`` / ``_tokenize_document``), mirroring how
    ``tools/preprocess_data.py --append-eod`` embeds boundaries into the
    offline binarized corpus.
    """
    seq_length = tokens.numel()

    attention_mask: torch.Tensor | None
    if create_attention_mask:
        attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=tokens.device)).unsqueeze(0)
    else:
        attention_mask = None

    loss_mask = torch.ones(seq_length, dtype=torch.float, device=tokens.device)
    if eod_mask_loss:
        loss_mask[tokens == eod_token] = 0.0

    position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        eod_index = position_ids[tokens == eod_token]
        if reset_position_ids:
            eod_index = eod_index.clone()

        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


class OnlinePretrainDataset(Dataset):
    """GPT-style pretraining dataset with on-the-fly tokenization.

    Mirrors Megatron-Core's offline GPT MMap dataset
    (``megatron.core.datasets.gpt_dataset.GPTDataset``): per split, documents
    are tokenized (EOD-terminated) and conceptually concatenated into one
    long token stream, with document order reshuffled every epoch. Fixed
    ``seq_length + 1`` windows are then sliced from that stream as samples
    (a window may span multiple, or partial, documents) and the usual
    next-token shift produces ``tokens``/``labels``.

    ``reset_position_ids``, ``reset_attention_mask``, and ``eod_mask_loss``
    have exactly the same semantics as ``GPTDatasetConfig``: boundaries are
    found by scanning the window for the EOD token value (see
    ``_build_ltor_mask_and_position_ids``); no cu_seqlens/THD packing is
    used.
    """

    def __init__(
        self,
        file_paths: list[str],
        split_start: int,
        split_end: int,
        tokenizer: MegatronTokenizer,
        seq_length: int,
        text_key: str = "text",
        append_eod: bool = True,
        reset_position_ids: bool = False,
        reset_attention_mask: bool = False,
        eod_mask_loss: bool = False,
        create_attention_mask: bool = True,
        seed: int = 1234,
        lengths_cache_path: str | None = None,
        num_dataset_builder_threads: int = 1,
        max_num_samples: int | None = None,
        file_offsets: list[int] | None = None,
    ) -> None:
        self.file_paths = file_paths
        self.split_start = split_start
        self.split_end = split_end
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_key = text_key
        self.append_eod = append_eod
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.create_attention_mask = create_attention_mask
        self.seed = seed
        self.num_dataset_builder_threads = num_dataset_builder_threads

        if file_offsets is not None:
            # Reuse offsets pre-computed once by build_train_valid_test_datasets
            # instead of re-scanning every file's metadata per split (train/
            # valid/test would otherwise each redo this over all files).
            self._file_offsets = file_offsets
        else:
            counts = count_records_batch(file_paths, num_workers=_scaled_builder_threads(num_dataset_builder_threads))
            self._file_offsets = [0]
            for c in counts:
                self._file_offsets.append(self._file_offsets[-1] + c)

        doc_lengths = self._compute_split_lengths(lengths_cache_path)
        num_docs = len(doc_lengths)
        total_tokens_per_epoch = sum(doc_lengths)
        if num_docs == 0 or total_tokens_per_epoch < 2:
            raise ValueError(
                f"OnlinePretrainDataset split [{split_start}, {split_end}) has too few tokens "
                f"({total_tokens_per_epoch}) to form a single seq_length={seq_length} sample."
            )

        # How many epochs' worth of (reshuffled) document order do we need to
        # cover max_num_samples? Mirrors GPTDataset._get_num_epochs.
        num_tokens_requested = (max_num_samples * seq_length + 1) if max_num_samples else total_tokens_per_epoch
        num_epochs = 1
        total = total_tokens_per_epoch
        while total < num_tokens_requested:
            num_epochs += 1
            total += total_tokens_per_epoch

        # Shuffle document order once per epoch and concatenate (mirrors
        # GPTDataset._build_document_index), then record cumulative token
        # offsets so __getitem__ can bisect into this virtual stream.
        rng = np.random.RandomState(seed)
        doc_order = np.concatenate([rng.permutation(num_docs) for _ in range(num_epochs)]).astype(np.int64)
        doc_lengths_arr = np.asarray(doc_lengths, dtype=np.int64)
        doc_cum_offsets = np.zeros(len(doc_order) + 1, dtype=np.int64)
        np.cumsum(doc_lengths_arr[doc_order], out=doc_cum_offsets[1:])
        self._doc_order = doc_order
        self._doc_cum_offsets = doc_cum_offsets

        num_samples = max(0, (int(doc_cum_offsets[-1]) - 1) // seq_length)
        if num_samples == 0:
            raise ValueError(
                f"OnlinePretrainDataset split [{split_start}, {split_end}) does not contain enough "
                f"tokens ({int(doc_cum_offsets[-1])}) to form a single seq_length={seq_length} sample."
            )

        # Sample-level shuffle so sequential idx (the Megatron samplers are
        # non-random) map onto random windows of the document stream
        # (mirrors GPTDataset._build_shuffle_index).
        sample_epochs = max(1, int(np.ceil((max_num_samples or num_samples) / num_samples)))
        sample_shuffle = np.concatenate([rng.permutation(num_samples) for _ in range(sample_epochs)])
        if max_num_samples is not None:
            sample_shuffle = sample_shuffle[:max_num_samples]
        self._sample_shuffle = sample_shuffle

        logger.info(
            "OnlinePretrainDataset: %d samples from %d docs (split [%d, %d), %d epoch(s) of "
            "document order, reset_position_ids=%s reset_attention_mask=%s)",
            len(self._sample_shuffle), num_docs, split_start, split_end, num_epochs,
            reset_position_ids, reset_attention_mask,
        )

    # ------------------------------------------------------------------
    # Length computation (rank-0 build + sync)
    # ------------------------------------------------------------------

    def _compute_split_lengths(self, cache_path: str | None) -> list[int]:
        cache_key = f"{cache_path}.{self.split_start}_{self.split_end}.npy" if cache_path else None
        num_workers = _scaled_builder_threads(self.num_dataset_builder_threads)
        return _rank0_build_and_sync(lambda: self._do_compute_lengths(num_workers), cache_key)

    def _do_compute_lengths(self, num_workers: int) -> list[int]:
        relevant: list[tuple[int, str]] = []
        for fidx, fp in enumerate(self.file_paths):
            if self._file_offsets[fidx + 1] > self.split_start and self._file_offsets[fidx] < self.split_end:
                relevant.append((fidx, fp))
        num_workers = max(1, min(num_workers, len(relevant)))

        results: dict[int, list[int]] = {}
        if num_workers <= 1:
            for fidx, fp in relevant:
                results[fidx] = _compute_file_lengths(fp, self.tokenizer, self.text_key, self.append_eod)
        else:
            try:
                from tqdm import tqdm as _tqdm
            except ImportError:
                _tqdm = None
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futs = {
                    pool.submit(_compute_file_lengths, fp, self.tokenizer, self.text_key, self.append_eod): fidx
                    for fidx, fp in relevant
                }
                it = as_completed(futs)
                if _tqdm:
                    it = _tqdm(it, total=len(futs), desc="Computing token lengths", unit="file", leave=False)
                for fut in it:
                    results[futs[fut]] = fut.result()

        flat: list[int] = []
        for fidx, _ in relevant:
            flat.extend(results[fidx])
        first_start = self._file_offsets[relevant[0][0]]
        return flat[self.split_start - first_start : self.split_end - first_start]

    # ------------------------------------------------------------------
    # Window gathering
    # ------------------------------------------------------------------

    def _tokenize_local_doc(self, local_doc_idx: int) -> list[int]:
        """Tokenize the document at *local_doc_idx* (0-based within this split)."""
        gidx = self.split_start + local_doc_idx
        fidx = bisect.bisect_right(self._file_offsets, gidx) - 1
        row = gidx - self._file_offsets[fidx]
        return _tokenize_document(self.file_paths[fidx], row, self.tokenizer, self.text_key, self.append_eod)

    def _gather_window(self, start: int, end: int) -> list[int]:
        """Collect ``end - start`` token ids from the shuffled document stream.

        Walks forward through ``self._doc_order`` starting from the document
        containing token offset *start*, slicing as many (partial) documents
        as needed. Pads the tail with ``_PAD_TOKEN_ID`` if the stream runs
        out (only possible for the very last sample of the split).
        """
        offsets = self._doc_cum_offsets
        pos = int(np.searchsorted(offsets, start, side="right")) - 1
        needed = end - start
        cursor = start
        tokens: list[int] = []
        while len(tokens) < needed and pos < len(self._doc_order):
            doc_start, doc_end = int(offsets[pos]), int(offsets[pos + 1])
            ids = self._tokenize_local_doc(int(self._doc_order[pos]))
            take_start = cursor - doc_start
            take_end = min(doc_end, end) - doc_start
            tokens.extend(ids[take_start:take_end])
            cursor = doc_start + take_end
            pos += 1
        if len(tokens) < needed:
            tokens.extend([_PAD_TOKEN_ID] * (needed - len(tokens)))
        return tokens

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sample_shuffle)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_id = int(self._sample_shuffle[idx])
        start = sample_id * self.seq_length
        end = start + self.seq_length + 1

        window = torch.LongTensor(self._gather_window(start, end))
        input_ids = window[:-1].contiguous()
        labels = window[1:].contiguous()

        attention_mask, loss_mask, position_ids = _build_ltor_mask_and_position_ids(
            input_ids,
            self.tokenizer.eod,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
            self.create_attention_mask,
        )

        # End-of-split padding: zero the loss and remap to a safe embeddable
        # id (mirrors megatron.core.datasets.megatron_dataset's handling of
        # _PAD_TOKEN_ID).
        is_pad = labels == _PAD_TOKEN_ID
        if bool(is_pad.any()):
            loss_mask[is_pad] = 0.0
            input_ids = torch.where(input_ids == _PAD_TOKEN_ID, torch.zeros_like(input_ids), input_ids)
            labels = torch.where(is_pad, torch.zeros_like(labels), labels)

        sample: dict[str, torch.Tensor] = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if attention_mask is not None:
            sample["attention_mask"] = attention_mask
        return sample

    def collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Stack fixed-length ``[L]`` samples into a dense ``[B, L]`` batch.

        No packing / cu_seqlens: every sample is already exactly
        ``seq_length`` tokens (mirrors the offline GPT MMap dataset's
        default batch shape).
        """
        result = {
            "tokens": torch.stack([item["tokens"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "loss_mask": torch.stack([item["loss_mask"] for item in batch]),
            "position_ids": torch.stack([item["position_ids"] for item in batch]),
        }
        if "attention_mask" in batch[0]:
            result["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_train_valid_test_datasets(
    file_paths: list[str],
    tokenizer: MegatronTokenizer,
    seq_length: int,
    split: str,
    text_key: str = "text",
    append_eod: bool = True,
    reset_position_ids: bool = False,
    reset_attention_mask: bool = False,
    eod_mask_loss: bool = False,
    create_attention_mask: bool = True,
    seed: int = 1234,
    lengths_cache_path: str | None = None,
    num_dataset_builder_threads: int = 1,
    train_samples: int = 0,
    valid_samples: int = 0,
    test_samples: int = 0,
) -> tuple[Optional[OnlinePretrainDataset], Optional[OnlinePretrainDataset], Optional[OnlinePretrainDataset]]:
    """Create train / valid / test datasets by splitting the global doc range."""
    # Compute file offsets once (parallel metadata reads) and share across
    # train/valid/test to avoid re-scanning every file 3 extra times.
    counts = count_records_batch(file_paths, num_workers=_scaled_builder_threads(num_dataset_builder_threads))
    file_offsets = [0]
    for c in counts:
        file_offsets.append(file_offsets[-1] + c)
    total_docs = file_offsets[-1]

    parts = [int(x) for x in split.split(",")]
    assert len(parts) == 3, f"split must have 3 comma-separated values: {split}"
    total_ratio = sum(parts)
    train_end = total_docs * parts[0] // total_ratio
    valid_end = train_end + total_docs * parts[1] // total_ratio

    def _make(start: int, end: int, n_samples: int) -> Optional[OnlinePretrainDataset]:
        if end <= start:
            return None
        return OnlinePretrainDataset(
            file_paths=file_paths,
            split_start=start,
            split_end=end,
            tokenizer=tokenizer,
            seq_length=seq_length,
            text_key=text_key,
            append_eod=append_eod,
            reset_position_ids=reset_position_ids,
            reset_attention_mask=reset_attention_mask,
            eod_mask_loss=eod_mask_loss,
            create_attention_mask=create_attention_mask,
            seed=seed,
            lengths_cache_path=lengths_cache_path,
            num_dataset_builder_threads=num_dataset_builder_threads,
            max_num_samples=n_samples if n_samples > 0 else None,
            file_offsets=file_offsets,
        )

    return (
        _make(0, train_end, train_samples),
        _make(train_end, valid_end, valid_samples),
        _make(valid_end, total_docs, test_samples),
    )
