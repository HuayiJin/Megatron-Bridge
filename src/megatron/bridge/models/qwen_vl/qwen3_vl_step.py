# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import math
import os
import socket
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.padding_utils import (
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)

_G_LOGGED_FIRST_BATCH_FETCH = False


def _summarize_batch_for_debug(batch: dict[str, Any]) -> str:
    summaries: list[str] = []
    for key in sorted(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            summaries.append(f"{key}=Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})")
        elif value is None:
            summaries.append(f"{key}=None")
        elif hasattr(value, "__dict__"):
            fields = []
            for field_name, field_value in sorted(value.__dict__.items()):
                if isinstance(field_value, torch.Tensor):
                    fields.append(f"{field_name}:shape={tuple(field_value.shape)},dtype={field_value.dtype}")
                elif field_value is not None:
                    fields.append(f"{field_name}:{type(field_value).__name__}")
            summaries.append(f"{key}={type(value).__name__}({', '.join(fields)})")
        else:
            summaries.append(f"{key}={type(value).__name__}")
    return "; ".join(summaries)


def _log_qwen3_vl_debug(stage: str, batch: dict[str, Any] | None = None) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = int(os.environ.get("RANK", "-1"))

    cuda_device = None
    memory_allocated_gb = 0.0
    memory_reserved_gb = 0.0
    if torch.cuda.is_available():
        cuda_device = torch.cuda.current_device()
        memory_allocated_gb = torch.cuda.memory_allocated() / 1024**3
        memory_reserved_gb = torch.cuda.memory_reserved() / 1024**3

    batch_summary = _summarize_batch_for_debug(batch) if batch is not None else ""
    logger.info(
        "[qwen3-vl-debug] stage=%s rank=%s local_rank=%s node_rank=%s host=%s cuda_device=%s "
        "memory_allocated_gb=%.2f memory_reserved_gb=%.2f batch=%s",
        stage,
        rank,
        os.environ.get("LOCAL_RANK", "unknown"),
        os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", "unknown")),
        socket.gethostname(),
        cuda_device,
        memory_allocated_gb,
        memory_reserved_gb,
        batch_summary,
    )


def _get_sample_seq_lens_for_log(batch: dict[str, Any]) -> list[int]:
    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return attention_mask.reshape(attention_mask.shape[0], -1).sum(dim=1).to(torch.int64).cpu().tolist()

    input_ids = batch.get("input_ids")
    if input_ids is None:
        input_ids = batch.get("tokens")
    if isinstance(input_ids, torch.Tensor):
        return [int(input_ids.shape[1])] * int(input_ids.shape[0])
    return []


def _print_sample_padding_truncation(sample_seq_lens: list[int], target_len: int, iteration: int) -> None:
    for sample_idx, sample_len in enumerate(sample_seq_lens):
        if sample_len < target_len:
            print_rank_0(
                f"[qwen3-vl-sample-length] iteration={iteration} sample_idx={sample_idx} "
                f"input_length={sample_len} target_length={target_len} padding_length={target_len - sample_len}"
            )
        elif sample_len > target_len:
            print_rank_0(
                f"[qwen3-vl-sample-length] iteration={iteration} sample_idx={sample_idx} "
                f"input_length={sample_len} target_length={target_len} truncation_length={sample_len - target_len}"
            )


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, Any]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.
        use_mtp: Whether Multi-Token Prediction layers are enabled.
        skip_getting_attention_mask_from_dataset: If set, the dataset will pass a None attention mask.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    global _G_LOGGED_FIRST_BATCH_FETCH

    should_log_batch_fetch = not _G_LOGGED_FIRST_BATCH_FETCH
    if should_log_batch_fetch:
        _log_qwen3_vl_debug("before-first-next-data-iterator")
    batch = next(data_iterator)
    sample_seq_lens = _get_sample_seq_lens_for_log(batch)
    if should_log_batch_fetch:
        _log_qwen3_vl_debug("after-first-next-data-iterator", batch)
        _G_LOGGED_FIRST_BATCH_FETCH = True

    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    # Instead of raw tensors, expect a single 'visual_inputs' object in batch
    required_device_keys.add("visual_inputs")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            if key == "visual_inputs":
                if val is None:
                    _batch_required_keys[key] = None
                else:
                    _batch_required_keys[key] = val
                    # Move all visual inputs contained tensors to CUDA
                    for k, v in val.__dict__.items():
                        _batch_required_keys[key].__dict__[k] = v.cuda(non_blocking=True) if v is not None else None
            else:
                _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    _batch_required_keys["sample_seq_lens"] = sample_seq_lens

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable,
    cfg: ConfigContainer,
    use_mtp: bool = False,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Any,
    list[int],
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled
        is_first_pp_stage: Whether the current stage is the first stage
        is_last_pp_stage: Whether the current stage is the last stage
    Returns:
        TODO: add description
    """
    # All PP stages load from iterator to get input_ids and visual grid info
    # This allows each stage to compute MRoPE position_ids locally without broadcasting
    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first_pp_stage,
        is_last_pp_stage=is_last_pp_stage,
    )

    if "visual_inputs" in batch and batch.get("visual_inputs") is not None:
        # convert visual_inputs to multi_modal_inputs which is a dict contains "pixel_values" and "image_grid_thw"
        # TODO(jinliangl): add video support
        multi_modal_inputs = batch.get("visual_inputs").normalized_for_model()
    else:
        multi_modal_inputs = {}

    # return naive batch and don't do any padding or cp slicing
    return (
        batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids"),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        multi_modal_inputs,
        batch.get("sample_seq_lens", []),
    )


def pack_or_pad_batch_sequences(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    this_pg_collection,
    use_fp8_padding: bool = False,
    force_to_pad_to_seq_len: bool = False,
    seq_length: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackedSeqParams]:
    """
    Pad or truncate the batch sequences to the target length, and build packed sequences.
    If is_qwen3vl, return bshd tokens for be compatible with qwen3vl model.
    Otherwise, return thd tokens and packed sequences.
    """

    batch_size, cur_len = tokens.shape
    device = tokens.device

    tp_size = this_pg_collection.tp.size()
    cp_size = this_pg_collection.cp.size()
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    divisible_by = math.lcm(divisible_by, 16) if use_fp8_padding else divisible_by

    # build bshd sequences with tiny padding to be compatible with qwen3vl model
    target_len = math.ceil(cur_len / divisible_by) * divisible_by
    if force_to_pad_to_seq_len:
        target_len = seq_length
    tokens = pad_or_truncate_2d_to_len(tokens, target_len=target_len, max_cap=target_len, pad_value=0)
    labels = pad_or_truncate_2d_to_len(labels, target_len=target_len, max_cap=target_len, pad_value=-100)
    loss_mask = pad_or_truncate_2d_to_len(loss_mask, target_len=target_len, max_cap=target_len, pad_value=0)
    attention_mask = pad_or_truncate_attn_to_len(attention_mask, target_len=target_len, max_cap=target_len)
    position_ids = pad_or_truncate_pos_to_len(position_ids, target_len=target_len, max_cap=target_len)

    seqlens_in_batch = torch.ones(batch_size, dtype=torch.int32, device=device) * target_len
    seqlens_in_batch_padded = torch.ones(batch_size, dtype=torch.int32, device=device) * target_len
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    this_pg_collection = get_pg_collection(model)
    is_first = is_pp_first_stage(this_pg_collection.pp)
    is_last = is_pp_last_stage(this_pg_collection.pp)

    config = get_model_config(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    if state.train_state.step == 0:
        _log_qwen3_vl_debug("forward-step-enter")
    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            multi_modal_inputs,
            sample_seq_lens,
        ) = get_batch(data_iterator, state.cfg, use_mtp, is_first_pp_stage=is_first, is_last_pp_stage=is_last)
    timers("batch-generator").stop()
    if state.train_state.step == 0:
        _log_qwen3_vl_debug("after-get-batch")

    # To be compatible with qwen3vl, we move the sequence padding and packing to forward_step function.
    # Qwen3VL model need the original input and do cp and sp split in model.forward.
    pack_sequences_in_batch = getattr(state.cfg.dataset, "pack_sequences_in_batch", False)
    tp_size = this_pg_collection.tp.size()
    cp_size = this_pg_collection.cp.size()
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    divisible_by = math.lcm(divisible_by, 16)
    target_len = math.ceil(tokens.shape[1] / divisible_by) * divisible_by
    if this_pg_collection.pp.size() > 1 or this_pg_collection.ep.size() > 1:
        target_len = config.seq_length
    _print_sample_padding_truncation(sample_seq_lens, target_len, state.train_state.step)

    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = pack_or_pad_batch_sequences(
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        this_pg_collection,
        use_fp8_padding=True,
        force_to_pad_to_seq_len=this_pg_collection.pp.size() > 1 or this_pg_collection.ep.size() > 1,
        seq_length=config.seq_length,
    )
    if state.train_state.step == 0:
        _log_qwen3_vl_debug(
            "after-pack-or-pad",
            {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
        )
    forward_args = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    original_tokens = tokens.clone()
    forward_args = get_batch_on_this_cp_rank(forward_args, cp_group=this_pg_collection.cp)
    forward_args["packed_seq_params"] = None
    forward_args["input_ids"] = original_tokens
    # calculate position_ids in model forward
    forward_args["position_ids"] = None
    if pack_sequences_in_batch:
        if forward_args["labels"] is not None:
            # When using pp, labels could be None
            forward_args["labels"] = forward_args["labels"].reshape(1, -1)
        attention_mask = torch.ones(
            original_tokens.shape[0], original_tokens.shape[1], dtype=torch.bool, device=original_tokens.device
        )
        forward_args["attention_mask"] = attention_mask
        if forward_args["loss_mask"] is not None:
            forward_args["loss_mask"] = forward_args["loss_mask"].reshape(1, -1)
        # qwen3vl need the original input_ids and position_ids
        # use split attention mask for calculate loss
        forward_args["packed_seq_params"] = packed_seq_params

    # use cp split loss mask for calculate loss
    loss_mask = forward_args["loss_mask"]
    # follow the design of verl, we put the multi-modal inputs in the forward args
    if "pixel_values" in multi_modal_inputs:
        forward_args["pixel_values"] = multi_modal_inputs["pixel_values"]
    if "image_grid_thw" in multi_modal_inputs:
        forward_args["image_grid_thw"] = multi_modal_inputs["image_grid_thw"]
    if "pixel_values_videos" in multi_modal_inputs:
        forward_args["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"]
    if "video_grid_thw" in multi_modal_inputs:
        forward_args["video_grid_thw"] = multi_modal_inputs["video_grid_thw"]

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
            return schedule_plan, loss_function
        else:
            if state.train_state.step == 0:
                _log_qwen3_vl_debug("before-model-forward")
            output_tensor = model(**forward_args)
            if state.train_state.step == 0:
                _log_qwen3_vl_debug("after-model-forward")

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
