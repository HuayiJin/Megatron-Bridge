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
import os
from functools import partial
from typing import Tuple

import torch
from megatron.core.rerun_state_machine import get_rerun_state_machine


_DEFAULT_SPIKY_LOSS_FACTOR: float = 10.0

# NaN/Inf diagnostics. Always-on lightweight per-rank fingerprint logging for the
# first few microbatches of training, plus full diagnostic dump whenever a NaN
# or Inf is detected. Tunable via env vars:
#   MBRIDGE_LOSS_DIAG_FIRST_N : log fingerprints for the first N loss_func calls
#                               on every rank (default 4 — covers iter 0/1 across
#                               a few PP microbatches).
#   MBRIDGE_LOSS_DIAG_VERBOSE : if set to "1", always log the full fingerprint,
#                               not just the first N calls.
# Deliberately uses print(..., flush=True) instead of logger.* so the lines
# survive even when the python logging config is misrouted.
_LOSS_DIAG_LOGGER = logging.getLogger(__name__)
_LOSS_DIAG_FIRST_N: int = int(os.environ.get("MBRIDGE_LOSS_DIAG_FIRST_N", "4"))
_LOSS_DIAG_VERBOSE: bool = os.environ.get("MBRIDGE_LOSS_DIAG_VERBOSE", "0") == "1"
_LOSS_DIAG_CALL_COUNT: int = 0


def _rank_tag() -> str:
    """Return a short rank identifier suitable for grep."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return f"rank{torch.distributed.get_rank():03d}"
    return "rank???"


def _tensor_fingerprint(name: str, t: torch.Tensor) -> str:
    """Cheap, NaN-safe summary of a tensor for diagnostic logging.

    Uses .float() to coerce bf16/fp16 into a printable scale; uses nansum / .any
    style ops where possible to avoid raising on degenerate inputs.
    """
    try:
        tf = t.detach().float()
        n_nan = torch.isnan(tf).sum().item()
        n_inf = torch.isinf(tf).sum().item()
        # min/max of the *finite* portion to avoid printing nan/inf as the extremes
        finite_mask = torch.isfinite(tf)
        n_finite = int(finite_mask.sum().item())
        if n_finite > 0:
            tf_finite = tf[finite_mask]
            fmin = tf_finite.min().item()
            fmax = tf_finite.max().item()
            fsum = tf_finite.sum().item()
            fmean = fsum / n_finite
        else:
            fmin = fmax = fsum = fmean = float("nan")
        return (
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"numel={t.numel()} n_nan={n_nan} n_inf={n_inf} n_finite={n_finite} "
            f"min={fmin:.4g} max={fmax:.4g} mean={fmean:.4g} sum={fsum:.4g}"
        )
    except Exception as e:  # noqa: BLE001 — diagnostic, never raise
        return f"{name}: <fingerprint failed: {type(e).__name__}: {e}>"


def create_masked_next_token_loss_function(
    loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
) -> partial:
    """Create a partial loss function configured for masked next-token loss.

    This replaces the generic helper previously in utils/loss_utils.py.
    """

    return partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def masked_next_token_loss(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor | Tuple[torch.Tensor],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Loss function.

    Args:
        loss_mask: Used to mask out some portions of the loss
        output_tensor: The tensor with the losses. For LLaVAModel, this is a tuple of (losses, new_loss_mask)
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        tuple containing:
        - The loss scalar for this micro-batch
        - The number of non-padded tokens in this microbatch
        - A dict containing reporting metrics on the loss and number of tokens across
          the data parallel ranks
    """
    output_was_tuple = isinstance(output_tensor, tuple)
    if output_was_tuple:
        losses = output_tensor[0].view(-1).float()
        # NOTE: when output_tensor is a tuple (LLaVA-style step), the second
        # element is the *new* loss_mask the model wants to use; this overrides
        # the loss_mask that was bound at create_masked_next_token_loss_function
        # time. We log both so a mismatch is visible.
        loss_mask_from_model = output_tensor[1].view(-1).float()
        loss_mask = loss_mask_from_model
    else:
        losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # ---- NaN / Inf diagnostic logging --------------------------------------
    # Always inspect the result. Three regimes:
    #   * first N calls (per rank) -> emit a one-line fingerprint regardless
    #   * env MBRIDGE_LOSS_DIAG_VERBOSE=1 -> emit fingerprint every call
    #   * NaN or Inf detected -> emit a *full* dump (losses + mask + product)
    #     before the rerun_state_machine.validate_result call raises.
    # All prints go to stderr with flush=True so they survive PyTorch's odd
    # buffering inside the dataloader / pipeline scheduler.
    global _LOSS_DIAG_CALL_COUNT
    _LOSS_DIAG_CALL_COUNT += 1

    loss_finite = bool(torch.isfinite(loss).item())
    losses_has_nan = bool(torch.isnan(losses).any().item())
    losses_has_inf = bool(torch.isinf(losses).any().item())
    abnormal = (not loss_finite) or losses_has_nan or losses_has_inf

    should_log_routine = _LOSS_DIAG_VERBOSE or _LOSS_DIAG_CALL_COUNT <= _LOSS_DIAG_FIRST_N
    if abnormal or should_log_routine:
        prefix = f"[loss-diag] {_rank_tag()} call#{_LOSS_DIAG_CALL_COUNT}"
        tag = "ABNORMAL" if abnormal else "ok"
        print(
            f"{prefix} {tag} "
            f"loss={loss.item() if loss.numel() == 1 else 'N/A'} "
            f"output_was_tuple={output_was_tuple} "
            f"loss_mask_sum={loss_mask.sum().item():.4g} "
            f"loss_mask_nonzero={int((loss_mask > 0).sum().item())}",
            flush=True,
        )
        if abnormal or _LOSS_DIAG_VERBOSE:
            # Full per-tensor fingerprint dump. Cheap (a few reductions) and
            # immensely useful to distinguish the failure modes:
            #   * losses has NaN          -> model forward emitted NaN
            #   * losses has Inf, mask=0  -> 0 * inf = NaN trap
            #   * loss_mask all zero      -> empty supervision (silently lost)
            print(f"{prefix}   {_tensor_fingerprint('losses', losses)}", flush=True)
            print(f"{prefix}   {_tensor_fingerprint('loss_mask', loss_mask)}", flush=True)
            if output_was_tuple:
                print(
                    f"{prefix}   note: model returned its own loss_mask via "
                    "tuple; the loss_mask bound at create_masked_next_token_loss_function "
                    "was overridden",
                    flush=True,
                )
            # Identify positions where losses is non-finite and check whether
            # they coincide with mask>0 (i.e. they actually pollute the sum).
            nonfinite_mask = ~torch.isfinite(losses)
            nonfinite_count = int(nonfinite_mask.sum().item())
            if nonfinite_count > 0:
                overlap = int((nonfinite_mask & (loss_mask > 0)).sum().item())
                print(
                    f"{prefix}   nonfinite_in_losses={nonfinite_count} "
                    f"overlap_with_mask_positive={overlap} "
                    f"(if overlap>0 the NaN propagates into the sum)",
                    flush=True,
                )
                first_idx = int(torch.where(nonfinite_mask)[0][0].item())
                print(
                    f"{prefix}   first_nonfinite_idx={first_idx} "
                    f"loss_mask_at_first_nonfinite={loss_mask[first_idx].item()}",
                    flush=True,
                )

    # ------------------------------------------------------------------------

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if check_for_spiky_loss:
        spiky_loss_factor = getattr(rerun_state_machine, "spiky_loss_factor", _DEFAULT_SPIKY_LOSS_FACTOR)
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=spiky_loss_factor,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {"lm loss": reporting_loss})
