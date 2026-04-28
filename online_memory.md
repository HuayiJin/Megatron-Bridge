# online_memory.md — Live Debug Log

> Running notebook of currently-active investigations. Each entry has a clear
> close-out condition; once resolved, distill the lesson into `memory.md` (and
> remove transient logs/diagnostics from the source tree).

---

## 2026-04-27 — `vision_mask` shape mismatch in Qwen3-VL forward

### Symptom

```
File ".../models/qwen_vl/modelling_qwen3_vl/model.py", line 524, in forward
    combined_embeddings[vision_mask] = vision_embeds
RuntimeError: shape mismatch: value tensor of shape [14726, 3072]
              cannot be broadcast to indexing result of shape [0, 3072]
```

Vision encoder produced 14,726 vision-token embeddings, but
`vision_mask = (input_ids == image_token_id)` is **all-False** in the batch.
=> data pipeline gave the model `pixel_values` without (enough) `<|image_pad|>`
placeholders in `input_ids`. Contract violation.

### Top hypotheses (ranked)

1. **Truncation removed `<|image_pad|>` tokens.**
   `model.seq_length=2048` ≪ 14,726 expected vision tokens for one large
   image; truncation in dataloader / labels shift / packed-seq path drops the
   placeholder run.
2. **`min_pixels` / `max_pixels` cap not honored by Qwen3VLProcessor.**
   `qwen2_5_collate_fn` is reused for `Qwen3VLProcessor`
   (collate.py mapping). Single image expanded to ~7,363 vision tokens (×2
   images → 14,726) means image was not downsized.
3. **`image_token_id` mismatch.** Bridge falls back to `151655`; Qwen3.5-VL
   processor may write a different id.
4. **Mixed with/without-image collate path** (`batch_with` + `batch_without`
   merge in `qwen2_5_collate_fn`) drops or misaligns fields.

### Diagnostics added (2026-04-27)

- `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py`
  Just before `combined_embeddings[vision_mask] = vision_embeds`:
  prints rank, `input_ids.shape`, `image_token_id`, `video_token_id`,
  `(input_ids == image_token_id).sum()`, `(input_ids == video_token_id).sum()`,
  `vision_mask.sum()`, `vision_embeds.shape`, `combined_embeddings.shape`,
  `image_grid_thw`. On mismatch dumps row-0 head/tail of `input_ids`.
  Tag: `[DIAG vision_mask]`.

- `src/megatron/bridge/data/vlm_datasets/collate.py`
  End of `qwen2_5_collate_fn` (before label/loss-mask construction):
  prints `batch_size`, `seq_len`, `image_pad_id` (from
  `processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")`),
  `#image_pad_in_input_ids`, `image_grid_thw`, `sum(prod(thw))` and
  `pixel_values.shape`. Tag: `[DIAG collate qwen2_5]`.

Both blocks are wrapped in `try/except` and only print — no behavior change.

### How to read the logs

After one forward step on rank with vision data, expect lines like:

```
[DIAG collate qwen2_5] batch_size=B seq_len=S image_pad_id=151655 \
  #image_pad_in_input_ids=N_pad image_grid_thw=[[t,h,w], ...] \
  sum(prod(thw))=P pixel_values.shape=(P_actual, ...)

[DIAG vision_mask] rank=R input_ids.shape=(B,S) \
  image_token_id=151655 (#in_input_ids=N_pad') ... \
  vision_mask.sum()=N_mask vision_embeds.shape=(N_ve, H) \
  combined_embeddings.shape=(B,S,H) image_grid_thw=[...]
```

Decision tree:

| Observation | Likely cause | Action |
|---|---|---|
| `N_pad == 0` in collate | id mismatch OR processor not expanding placeholders | check `image_pad_id` matches model `image_token_id`; check Qwen3VL `apply_chat_template` output |
| `N_pad > 0` in collate but `N_pad' == 0` in model | truncation between collate and model forward; or `image_token_id` differs from `image_pad_id` | grep dataloader for truncation; reconcile ids |
| `N_pad > 0` and equals `N_ve` but `vision_mask.sum() == 0` | `vision_mask` recomputed from wrong id (bug in `reorganize_inputs`) | fix mask construction |
| `N_pad < N_ve` (current case: 0 vs 14726) | truncation OR image too large | reduce `max_pixels` / increase `seq_length` / pre-resize images |
| `image_grid_thw` huge (`prod` ≫ seq_length) | `min_pixels`/`max_pixels` not honored by Qwen3VLProcessor | switch to processor-native cap kwargs |

### Close-out checklist

- [ ] Run one forward; collect both `[DIAG collate qwen2_5]` and `[DIAG vision_mask]` lines.
- [ ] Identify root cause via decision tree above.
- [ ] Apply fix (cap `max_pixels` / pre-resize / fix id / fix mask).
- [ ] Remove the two `[DIAG ...]` print blocks from source.
- [ ] Add lesson + Pitfall # entry to `memory.md`.

### Notes / Constraints

- Do NOT enable `pack_sequences_in_batch` (memory.md: GDN/linear-attention is
  BSHD-only).
- 122B run is on TP=2 PP=4 EP=4 DP=4, 32 GPU, `seq_length=2048`,
  `pack_sequences_in_batch=False`.
- Processor: `Qwen3VLProcessor` (collate routed via `qwen2_5_collate_fn`).
