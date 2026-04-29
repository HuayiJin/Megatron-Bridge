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

"""
Core dataset types for conversation-style VLM examples.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional

import torch

from megatron.bridge.data.vlm_datasets.collate import COLLATE_FNS


class VLMConversationDataset(torch.utils.data.Dataset):
    """Repeating wrapper over a list of HF-style conversation examples.

    - Each base example is expected to contain a "conversation" key following
      processor.apply_chat_template conventions. Optional modality fields like
      "audio" are passed through and consumed by the collate function.
    - Dataset length is set to a target length and indexes wrap around the
      underlying list to meet the requested size.
    - A `collate_fn` attribute is exposed so the framework can pass it to the
      DataLoader.
    """

    def __init__(
        self,
        base_examples: List[Dict[str, Any]],
        target_length: int,
        processor: Any,
        collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None,
        *,
        max_length: Optional[int] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            base_examples: Non-empty list of conversation dicts.
            target_length: Virtual dataset length (wraps around base_examples).
            processor: HuggingFace processor used to tokenise examples.
            collate_impl: Optional override for the collate function. When
                ``None`` the implementation is chosen from ``COLLATE_FNS``
                based on the processor type name.
            max_length: Hard truncation limit (in tokens) forwarded to the
                collate function. Should equal ``dataset.seq_length`` so that
                sequences exceeding the model's context window are silently
                truncated in the DataLoader rather than causing PP shape
                mismatches at runtime (Pitfall #23).
        """
        assert isinstance(base_examples, list) and len(base_examples) > 0, "base_examples must be a non-empty list"
        self._base_examples = base_examples
        self._length = int(max(0, target_length))
        self._processor = processor
        self._max_length = max_length
        # Choose collate implementation by processor type name when not provided
        collate_key = type(processor).__name__ if processor is not None else "default"
        selected_impl = collate_impl or COLLATE_FNS.get(collate_key, COLLATE_FNS["default"])  # type: ignore[index]

        # Determine at construction time whether the chosen collate function
        # accepts a ``max_length`` keyword argument.  This avoids a TypeError
        # when ``max_length`` is set but the collate implementation does not
        # declare the parameter (e.g. glm4v_collate_fn, default_collate_fn).
        _collate_accepts_max_length = "max_length" in inspect.signature(selected_impl).parameters
        _max_length = max_length  # capture for closure

        def _bound_collate(batch: list) -> Dict[str, torch.Tensor]:
            if _max_length is not None and _collate_accepts_max_length:
                return selected_impl(batch, self._processor, max_length=_max_length)  # type: ignore[call-arg]
            return selected_impl(batch, self._processor)  # type: ignore[call-arg]

        self.collate_fn = _bound_collate

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        base = self._base_examples[idx % len(self._base_examples)]
        return base
