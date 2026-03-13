# Face Edit Feature — Design Spec
**Date:** 2026-03-13
**Project:** EchoFace

---

## Overview

After a face is generated, users can make targeted edits to specific attributes (eye colour, hair colour, hairstyle, etc.) using the same voice-or-text input flow as generation. Edits are cumulative — each edit starts from the current face's W+ latent — and a Reset button restores the original generated face at any time.

---

## User Flow

1. User generates a face (existing flow, unchanged)
2. **Edit Details section appears** in the left panel below the Generate button
3. User records audio or types an edit instruction (e.g. *"change eyes to blue"*, *"make the hair shorter and black"*)
4. User clicks **Apply Edit**
   - If audio provided with no text: transcribes, fills text box, waits for second click (same two-stage ASR pattern as generation)
   - If text present: runs edit immediately
5. Optimizer runs 60 steps from the current W+ latent with a combined prompt
6. Updated face replaces the image in the output panel
7. User can apply further edits (cumulative) or click **Reset** to restore the original

---

## Architecture

### Backend — `generator/styleclip_wrapper.py`

**State fields added to `StyleCLIPWrapper.__init__`:**
- `self._current_w: torch.Tensor | None = None` — W+ latent of the most recently generated or edited face
- `self._original_w: torch.Tensor | None = None` — W+ latent from the first `generate()` call; never overwritten by `edit()`
- `self._original_prompt: str | None = None` — the **raw user-supplied prompt string**, stored *before* the `"a photo of a face, "` prefix is applied, so `edit()` can reconstruct the combined prompt cleanly
- `self._original_sim: float | None = None` — CLIP similarity score from the original `generate()` call; returned by `reset()` without recomputation

**Modified `generate(text_prompt, ...)`:**
- Stores `self._original_prompt = text_prompt` (raw, pre-prefix)
- After optimization completes, stores `self._current_w = w_opt.detach().clone()` and `self._original_w = self._current_w.clone()`
- Stores `self._original_sim = final_sim`
- All existing return behaviour unchanged

**New method — `edit(edit_instruction, steps=60, lr=0.05, callback=None) -> (PIL.Image, float)`:**
- Raises `RuntimeError("Generate a face first before editing.")` if `_current_w is None`
- Constructs combined prompt: `f"a photo of a face, {self._original_prompt}, {edit_instruction}"`
- Initialises `w_opt` from `_current_w.clone().requires_grad_(True)` (not `w_avg`)
- L2 anchor: `_current_w.detach()` — regulariser keeps the face stable around current state (not `w_avg`, unlike `generate()`)
- L2 lambda: `0.05` (vs. `generate()`'s `0.02` — stronger anchor to preserve face identity during edits)
- All other hyperparameters identical to `generate()`: 60 steps, lr=0.05, cosine LR, early stopping
- After optimization: `self._current_w = w_opt.detach().clone()` (cumulative — next edit starts here)
- Returns `(PIL.Image, float)` — same signature as `generate()`

**New method — `reset() -> (PIL.Image, float)`:**
- Raises `RuntimeError("Generate a face first.")` if `_original_w is None`
- Copies: `self._current_w = self._original_w.clone()`
- Single synthesis forward pass (no optimization) to reconstruct original face image
- Returns `(PIL.Image, self._original_sim)` — CLIP score comes from cached `_original_sim`, not recomputed

---

### Frontend — `ui/app.py`

**Yield tuple:** `generate_fn` yield grows from 7 to **8 elements**. The new 8th element controls edit section visibility:

```
(status, text_in, score, image_arr, stored_pil, export_btn, loader_html, edit_section)
```

- All non-done yields: `edit_section = gr.update()` (no change)
- Done yield only: `edit_section = gr.update(visible=True)`
- Gradio outputs list for `gen_btn.click` updated to include `edit_section_col` as 8th output

**New function — `edit_fn(mic_audio, edit_text, wrapper, cfg)`:**
- Signature: **2 audio args** — `mic_audio` only (no file upload in edit flow; keep the edit panel minimal)
- Mirrors `generate_fn` structure: ASR stage → guard → optimization stage
- Short-prompt guard: `< 4 words` (aligned with `generate_fn`)
- Guard message: *"⚠ Edit too short — be specific (e.g. 'change eyes to blue')"*
- Stage 1 ASR status: *"Transcribing... Review and click Apply Edit again."*
- Stage 2 calls `wrapper.edit(edit_instruction, callback=_cb)`
- Yields **8-element tuple** (same schema as updated `generate_fn`):
  `(status, edit_text, score, image_arr, stored_pil, apply_btn_update, loader_html, gr.update())`
  - Element 5 (`apply_btn_update`): `gr.update(visible=False)` during progress, `gr.update(visible=True)` on done
  - Element 7 (`edit_section`): always `gr.update()` — visibility already set, no change needed
- `stored_pil` (element 4) updated with new PIL image so Export PNG always exports the latest face

**New function — `reset_fn(wrapper)`:**
- Calls `pil_image, sim = wrapper.reset()`
- Returns **5-element tuple**: `(np.array(pil_image), "Reset to original.", f"CLIP Score: {sim:.4f}", pil_image, gr.update(visible=True))`
  - Outputs: `[image_out, status_out, score_out, stored_image, export_btn]`
  - `stored_image` updated so Export PNG after Reset correctly exports the original face

**Thread safety:** `demo.queue(concurrency_count=1)` already serialises all events — only one of generate/edit/reset runs at a time. No additional locking needed. Apply Edit and Reset buttons are naturally blocked while Gradio's queue is processing.

**UI additions in `_build_demo()`:**

```python
with gr.Column(visible=False, elem_id="edit-section") as edit_section_col:
    gr.HTML('<div class="ef-section-label ef-edit-label">Edit Details</div>')
    edit_mic_in  = gr.Audio(source="microphone", type="filepath", label="Microphone")
    edit_text_in = gr.Textbox(lines=2, label="Edit Instruction",
                              placeholder="e.g.  change eyes to blue")
    with gr.Row():
        apply_btn = gr.Button("Apply Edit", elem_id="apply-btn")
        reset_btn = gr.Button("Reset",      elem_id="reset-btn")
```

Wiring:
```python
apply_btn.click(
    fn=functools.partial(edit_fn, wrapper=wrapper, cfg=cfg),
    inputs=[edit_mic_in, edit_text_in],
    outputs=[status_out, edit_text_in, score_out, image_out,
             stored_image, apply_btn, loader_html, edit_section_col],
)
reset_btn.click(
    fn=lambda: reset_fn(wrapper),
    inputs=[],
    outputs=[image_out, status_out, score_out, stored_image, export_btn],
)
```

**CSS additions:**
```css
#edit-section { border-top: 1px solid #1a3a2a !important; margin-top: 0.8rem !important; padding-top: 0.8rem !important; }
.ef-edit-label { color: #00ff99 !important; }
#apply-btn { background: linear-gradient(135deg, #0a6a3a 0%, #00884a 100%) !important; border: none !important; ... }
#reset-btn { background: transparent !important; border: 1px solid #2e4a6a !important; color: #5a7a9a !important; ... }
```

---

## Data Flow Summary

```
generate() called
  ├─ stores _original_w, _current_w, _original_prompt, _original_sim
  └─ edit section becomes visible in UI

edit() called
  ├─ reads _current_w as starting latent
  ├─ builds: "a photo of a face, {_original_prompt}, {edit_instruction}"
  ├─ runs 60-step optimization (L2 anchored to _current_w, λ=0.05)
  ├─ updates _current_w with result
  └─ returns (PIL.Image, float)

reset() called
  ├─ copies _original_w → _current_w
  ├─ single synthesis forward pass
  └─ returns (original PIL.Image, _original_sim)
```

---

## Error Handling

| Situation | Response |
|---|---|
| `edit_fn` called before `generate_fn` | `wrapper.edit()` raises → UI shows *"Generate a face first before editing."* |
| Empty edit instruction + no audio | *"Please provide an edit instruction or record audio."* |
| Edit instruction < 4 words | *"⚠ Edit too short — be specific (e.g. 'change eyes to blue')"* |
| Optimizer NaN | Same step-skip guard as `generate()` |
| `reset_fn` called before `generate_fn` | `wrapper.reset()` raises → UI shows *"Generate a face first."* |

---

## Out of Scope

- Undo history beyond one-level Reset
- Edit strength/intensity slider
- Side-by-side before/after view
- Per-attribute dropdowns

---

## Files Changed

| File | Change |
|---|---|
| `generator/styleclip_wrapper.py` | Add 4 state fields; add `edit()` and `reset()` methods; update `generate()` to store state |
| `ui/app.py` | Add `edit_fn()`, `reset_fn()`; add edit UI column; update `generate_fn` to 8-element yield; add CSS for edit section and buttons |
