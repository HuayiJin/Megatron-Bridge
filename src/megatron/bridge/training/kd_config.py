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

"""KD fork: offline cached-logits knowledge distillation configuration.

Two mutually compatible modes driven by one config container:

* **Teacher dump** — set ``save_logits_dir``. ``LogitsSaverHooks`` is attached
  to the model; each training iteration's top-K teacher log-probs are buffered
  and periodically flushed to batched tars under ``save_logits_dir``. The LM
  loss is zeroed out (gradient edges preserved); run with lr=0 to keep weights
  frozen.
* **Student KD** — set ``logprobs_dir``. ``StudentLogitsCapture`` is attached
  and the training loss becomes ``(1 - alpha) * LM + alpha * KL(teacher || student)``
  with alpha linearly interpolated from ``kd_alpha_start`` to ``kd_alpha_end``
  over ``kd_alpha_total_iters`` (SlimQwen recipe: 1.0 -> 0.75).

Teacher and student runs MUST use identical data pipeline settings (seed,
blend, sequence length, global batch size, micro batch size) so the cached
iteration payloads align with the student's sample stream. The tar metadata
carries a dataset-identity hash verified on the student side at load time.
"""

from dataclasses import dataclass, field
from typing import Optional

from megatron.bridge.training.utils.config_utils import _ConfigContainerBase as Container


@dataclass
class KDCachedLogitsConfig(Container):
    """Configuration for offline cached-logits knowledge distillation."""

    # ------------------------- teacher dump mode -------------------------
    save_logits_dir: Optional[str] = None
    """Directory for teacher top-K log-prob tars. Setting this enables
    LogitsSaverHooks on the model (teacher dump mode)."""

    save_top_k: int = 20
    """Number of top log-probabilities kept per token position."""

    save_top_p: Optional[float] = None
    """Optional nucleus truncation (0 < p <= 1). None keeps a fixed top-K."""

    save_top_p_min_k: int = 1000
    """Minimum K retained when top-p truncation is active."""

    save_dtype: str = "fp16"
    """Storage dtype for log-prob values: one of fp16/bf16/fp32."""

    dump_forward_only: bool = True
    """Teacher dump mode: skip backward + optimizer step (forward-only pipeline
    schedule, same data consumption order as training). Validated byte-identical
    against a full-step dump on iterations 0-160 (see kd/scripts/compare_dumps.py)."""

    flush_interval_iters: int = 10
    """Teacher mode: flush pending payloads to a tar every N iterations."""

    # -------------------------- student KD mode --------------------------
    logprobs_dir: Optional[str] = None
    """Directory holding teacher tars. Setting this enables the KD loss term
    (student mode). Requires the teacher tars to have been produced with the
    same data pipeline configuration."""

    kd_alpha_start: float = 1.0
    """KD weight lambda at iteration 0 (SlimQwen: starts at pure KD)."""

    kd_alpha_end: float = 0.75
    """KD weight lambda at kd_alpha_total_iters (SlimQwen linear decay)."""

    kd_alpha_total_iters: Optional[int] = None
    """Iteration at which alpha reaches kd_alpha_end. Defaults to
    train.train_iters when unset."""

    kd_decode_threads: int = 1
    """Student mode: CPU decode threads for tar payload decompression."""

    kd_prefetch_factor: int = 2
    """Student mode: DataLoader prefetch factor for tar streaming."""

    kd_ignore_errors: bool = False
    """Student mode: fall back to pure LM loss on KD payload errors instead
    of raising. Useful for smoke tests; leave False in production."""

    kd_start_iteration: int = 0
    """Student mode: first teacher iteration to consume (resume support)."""

    def finalize(self) -> None:
        """Validate KD configuration."""
        if self.save_logits_dir is None and self.logprobs_dir is None:
            return
        if self.save_logits_dir is not None:
            assert self.save_top_k > 0, f"save_top_k must be > 0, got {self.save_top_k}"
            assert self.flush_interval_iters > 0, (
                f"flush_interval_iters must be > 0, got {self.flush_interval_iters}"
            )
        if self.logprobs_dir is not None:
            assert 0.0 <= self.kd_alpha_start <= 1.0, "kd_alpha_start must be in [0, 1]"
            assert 0.0 <= self.kd_alpha_end <= 1.0, "kd_alpha_end must be in [0, 1]"
        assert not (self.save_logits_dir is not None and self.logprobs_dir is not None), (
            "Teacher dump (save_logits_dir) and student KD (logprobs_dir) are "
            "separate runs; enable only one per process."
        )
