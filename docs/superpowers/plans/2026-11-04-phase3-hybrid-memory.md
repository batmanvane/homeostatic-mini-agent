# Phase 3 Hybrid Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage S0 of the hybrid design: a conversation-mode agent session with four memory conditions and state sketches, a replay trainer with minibatch 1, an evaluation script, configs and a desktop chain, all tested on the Mac with a random tiny Qwen3.5.

**Architecture:** `homeo.memory.Session` wraps an `LM` and owns the cache per step (`none`/`full`/`state`/`kv`), returns the digit distribution with or without gradient, and commits the sampled digit into the memory. `run_episode_mem` plays an episode through a Session; the trainer replays logged episodes through a fresh `full` Session with gradients, detaching the cache after every step. Scripts expose `train(cfg, lm)` / `evaluate(cfg, make_lm)` so the tests run them on the tiny model.

**Tech Stack:** torch 2.15, transformers 5.22.1 (`Qwen3_5ForCausalLM`, `DynamicCache` with `get/update_recurrent_state`, `get/update_conv_state`), peft 0.21.2, bitsandbytes on the desktop only, pytest.

**Spec:** `docs/JOURNAL.md` (the README until 20.11.2026), section "Phase 3 hybrid design: memory in the loop, two timescales of learning (04.11.2026)".

## Global Constraints

- Agent never gets tools; world unchanged (`homeo.world`), v3 world config, reward v3b (task_bonus 0.02).
- Device-independent: tests run on CPU with a random tiny Qwen3.5 (config built from `Qwen3_5TextConfig`, tokenizer `Qwen/Qwen3.5-9B` from the HF cache, `HF_HUB_OFFLINE=1`); the 9B model is only loaded on the desktop (8-bit, never `prepare_model_for_kbit_training`).
- Continuation text per step: `f"{digit}<|im_end|>\n" + user_fragment`; the conversation is never re-rendered.
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, in_proj_qkv, out_proj`; r 8, alpha 16.
- Memory modes: `none`, `full`, `state`, `kv` exactly as the spec defines them (own action inside every carried memory).
- Sketch: 256 fixed Gaussian directions, seed 0, shared across layers, float16 output.
- Logged step rows keep the v3 keys (`analyze_v3_conditional.py` must still run on them) plus `memory`, `user_text` (training only), `shadow_none` probs.

## Review Focus

1. Replay must reproduce the rollout distribution (spec: within 1e-3): pinned by `test_replay_matches_rollout` (Task 4).
2. The cache must not keep the autograd graph across steps (memory blow-up on the 9B): pinned by `test_detach_cache` (Task 2).
3. In `kv`, the DeltaNet states must be zero when the user turn starts, after the answer tokens: pinned by `test_kv_zeroes_state_after_answer` (Task 2).
4. An episode that ends by death or capture must still commit nothing beyond the last step and the trainer must handle episodes of different lengths: pinned by `test_train_dry_run` with max_steps 3 and group 2 (Task 4).
5. Hidden-state logging must not hold 33 × 4096 per step for 36k steps: `hidden_layers` subsetting pinned by `test_session_hidden_layers` (Task 2).

---

### Task 1: Prompt fragments for conversation mode

**Files:**
- Modify: `src/homeo/prompts.py` (append functions)
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `render_system(cfg: PromptConfig) -> str` = `f"{INTRO} {FRAMING[cfg.framing]}"`; `render_user(world: World, menu: list[Action], cfg: PromptConfig) -> str` = status block (without the two leading newlines) + `"\n\nWhat do you do next? Options:\n{options}\n\n{instruction}"`, i.e. `render_prompt` minus the system part; `user_fragment(tokenizer, user_text: str) -> str` = the template suffix for one more user turn plus generation prompt, computed as `render([sys, u, a, user_text], gen) [len(render([sys, u, a], no gen)):]` with dummy strings and `enable_thinking=False`; `answer_text(digit: int) -> str` = `f"{digit}<|im_end|>\n"`.

- [x] **Step 1: Write the failing tests** in `tests/test_memory.py` with a module fixture `tok` (skip if `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")` fails offline):

```python
def test_fragment_extends_prefix_exactly(tok):
    sys_t, u1, u2 = "S", "first", "second"
    step1 = tok.apply_chat_template([{"role": "system", "content": sys_t}, {"role": "user", "content": u1}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cont = answer_text(2) + user_fragment(tok, u2)
    full = tok.apply_chat_template([... sys, u1, assistant "2", u2 ...], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # the template drops the think block of past assistant turns; our continuation keeps it
    assert tok(step1 + cont, add_special_tokens=False).input_ids[: len(tok(step1, add_special_tokens=False).input_ids)] == tok(step1, add_special_tokens=False).input_ids
    assert cont.startswith("2<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n")
    assert full.endswith(user_fragment(tok, u2))

def test_render_user_matches_render_prompt(world, cfg):  # world: World(WorldConfig()) with refreshed messages
    assert render_system(cfg) + "\n\n" + render_user(world, menu, cfg) == render_prompt(world, menu, cfg)
```

- [x] **Step 2: Run** `uv run pytest tests/test_memory.py -q` → FAIL (ImportError).
- [x] **Step 3: Implement** the four functions in `src/homeo/prompts.py`; `render_user` reuses `status_block` with the leading `"\n\n"` stripped (visibility `none` gives an empty block, then no leading newlines).
- [x] **Step 4: Run** the tests → PASS.
- [x] **Step 5: Commit** `feat(prompts): conversation-mode fragments`.

### Task 2: Session with four memory modes and state sketches

**Files:**
- Create: `src/homeo/memory.py`
- Modify: `src/homeo/model.py` (`LM.from_model` classmethod, `enable_thinking=False` in `_encode`, `bits` argument)
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: Task 1 functions; `LM` (`.model`, `.tokenizer`, `.device`, `.digit_ids`, `.n_layers`).
- Produces:
  - `LM.from_model(model, tokenizer, device: str) -> LM` (no loading; sets digit_ids, n_layers, hidden_size); `LM.__init__(name, device=None, dtype=None, bits: int | None = None)` — `bits == 8` → `BitsAndBytesConfig(load_in_8bit=True)`, `device_map=device`, no `.to()`; `load_lm(..., bits=None)` passes it through.
  - `MEMORY_MODES = ("none", "full", "state", "kv")`.
  - `class StateSketch(dim=256, seed=0)`: `__call__(states: dict[int, torch.Tensor]) -> np.ndarray` shape `(len(states), dim)` float16; projection matrix `(numel, dim)` built lazily on the first call for the state numel, dtype of the state, device of the state, `torch.Generator` seeded with `seed`, entries N(0, 1/dim).
  - `@dataclass StepOut: probs: np.ndarray; mass: float; hidden: np.ndarray | None; sketch: np.ndarray | None; logprobs: torch.Tensor | None` (`logprobs` only when `grad=True`, shape `(n_options,)`, renormalized over the menu digits as in `option_logprobs`).
  - `class Session(lm, mode: str, system_text: str, sketch: StateSketch | None = None, hidden_layers: list[int] | None = None)`:
    - `step(user_text: str, n_options: int, grad: bool = False) -> StepOut`: builds the input ids (first step or `none`/`state`: full template of system + user with generation prompt; later `full`/`kv` steps: `user_fragment` only), prepares the cache per mode (`none`: no cache; `state`: `DynamicCache(config=model.config text config)` + `update_recurrent_state(s, i)` / `update_conv_state(c, i)` for every linear layer from `self.carried`; `kv`: zero all recurrent and conv states of `self.cache` in place; `full`: `self.cache`), runs the model with `use_cache=True` (and `output_hidden_states=True` when `hidden_layers`), stores `self.cache`, returns probs/mass/hidden (rows `hidden_layers` of hidden_states at the last token, float16) /sketch of the linear layers' recurrent states.
    - `commit(digit: int) -> None`: `none`: no-op; otherwise run `answer_text(digit)` tokens (no grad, `add_special_tokens=False`) through `self.cache`; `state`: then copy the recurrent and conv states into `self.carried` (detached clones).
    - `detach() -> None`: detach every tensor in `self.cache` in place (DynamicLayer `keys`/`values`; LinearAttentionLayer `recurrent_states[k]`/`conv_states[k]`).
    - `linear_layers: list[int]` = indices with `layer_types == "linear_attention"`; `n_tokens: int` = cache sequence length (0 for `none`).
  - `run_episode_mem(world, lm, rng, prompt_cfg, mode, max_steps, sketch=None, hidden_layers=None, shadow_none=False, keep_text=False) -> list[dict]`: like `intero.run_episode` without the channel; rows have the v3 keys plus `memory`, `shadow_none` (probs dict of the same prompt as a fresh single turn, or None), `n_tokens`; `keep_text` adds `user_text`; returns rows; hidden and sketch arrays are returned as `row["hidden"]` / `row["sketch"]` (popped by the callers before logging). Work answers via `lm.answer(task_prompt(world))` as before.

- [x] **Step 1: Write the failing tests** (fixture `tiny_lm`: `Qwen3_5TextConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, intermediate_size=128, vocab_size=len(tok), layer_types=[lin, lin, lin, full]`, `torch.manual_seed(0)`, `AutoModelForCausalLM.from_config`, float32, `LM.from_model(model, tok, "cpu")`):

```python
def test_full_matches_single_pass(tiny_lm):      # 3 steps in full mode == one forward over the whole text
def test_state_equals_none_on_first_step_and_differs_later(tiny_lm)
def test_kv_equals_full_on_first_step_and_differs_later(tiny_lm)
def test_kv_zeroes_state_after_answer(tiny_lm):  # after step 2 in kv, the recurrent state equals the state of a
                                                 # fresh pass over the user fragment alone (states reset, KV kept)
def test_state_carries_own_answer(tiny_lm):      # state after commit(2) != state after commit(3)
def test_detach_cache(tiny_lm):                  # grad=True -> some cache tensor has grad_fn; detach() -> none
def test_sketch_is_linear_and_fixed(tiny_lm):    # shape (3, 256); sketch(a+b) == sketch(a)+sketch(b); same seed same result
def test_session_hidden_layers(tiny_lm):         # hidden_layers=[0, 4] -> hidden shape (2, 64)
def test_run_episode_mem_rows(tiny_lm):          # 3 steps, mode full, shadow_none: keys present, probs sum to 1
```

- [x] **Step 2: Run** → FAIL (no module `homeo.memory`).
- [x] **Step 3: Implement** `src/homeo/memory.py` and the `model.py` changes. Note: `DynamicCache(config=...)` needs the text config (`getattr(model.config, "text_config", model.config)`); `update_recurrent_state(tensor, layer_idx)` takes the tensor first.
- [x] **Step 4: Run** `uv run pytest tests/test_memory.py -q` → PASS; `uv run ruff check src tests` clean.
- [x] **Step 5: Commit** `feat(memory): session with none/full/state/kv memory and state sketches`.

### Task 3: RunLog sketches and LoRA target check

**Files:**
- Modify: `src/homeo/runlog.py` (`step(record, hidden=None, sketch=None)`, saves `sketch` array in `activations.npz`)
- Test: `tests/test_memory.py::test_runlog_sketch`

- [x] **Step 1: Test:** log two rows with hidden `(2, 8)` and sketch `(3, 4)`; after `close()` the npz has `hidden` `(2, 2, 8)` and `sketch` `(2, 3, 4)`.
- [x] **Step 2: Run** → FAIL (unexpected keyword).
- [x] **Step 3: Implement** (a second list; `np.savez_compressed(..., hidden=..., sketch=...)` only for non-empty lists).
- [x] **Step 4: Run** → PASS. **Step 5: Commit** `feat(runlog): sketch arrays`.

### Task 4: Replay trainer

**Files:**
- Create: `scripts/train_homeo_mem.py`
- Create: `configs/train_homeo_hy_s21.yaml`, `configs/train_homeo_hy_shuffle_s21.yaml`, `configs/train_homeo_hy_dry.yaml`
- Test: `tests/test_memory.py::test_replay_matches_rollout`, `::test_train_dry_run`

**Interfaces:**
- Consumes: `Session`, `run_episode_mem`, `learn.step_reward/discounted_returns/group_advantages`, `RunLog`.
- Produces: `train(cfg: dict, lm: LM, out_dir: Path) -> Path` (adapter must already be attached to `lm.model` by the caller when `cfg["lora"]` is used; `train` attaches it itself when `cfg.get("lora")` and the model is not a `PeftModel`), `main(config_path)`. Config keys as v3b plus `memory: full`, `dry_run: false`, `hidden_layers: [0, 8, 16, 24, 32]`, `sketch_dim: 256`, `bits: 8`, `lr: 2.0e-5`, `iterations: 16`, `checkpoint_every: 4`, `group_size: 8`, `max_steps: 150`, lora targets per Global Constraints.
- Replay: per episode a new `Session(lm, "full", system_text)`; per step `out = session.step(row["user_text"], len(row["menu"]), grad=True)`; `loss = -(adv * out.logprobs[row["choice_idx"]]) / n_steps_total`; `loss.backward()`; `session.detach()`; `session.commit(row["choice_idx"] + 1)`. Iteration 0 self-check: `max |exp(out.logprobs) − row["probs"]|` logged as `replay_max_dev` in the iteration summary; the gradient norm before clipping as `grad_norm`; in addition `grad_norm_shuffled`: same replay with permuted advantages is NOT run (too expensive); instead log `adv_abs_mean` per action (mean advantage of the steps that chose each action) and `adv_perm_band` (2.5/97.5 percentiles of that mean over 1000 permutations of the advantages, from `learn.permutation_band(adv, chosen_mask, n_perm, rng)` – new pure-numpy function in `learn.py`, tested with a constant advantage vector giving a zero-width band). `dry_run`: everything except `opt.step()`; checkpoints skipped.

- [x] **Step 1: Tests:**

```python
def test_permutation_band():  # learn.permutation_band(np.ones(10), mask, 100, rng) == (1.0, 1.0)
def test_replay_matches_rollout(tiny_lm):  # episode via run_episode_mem(full, keep_text) then replay with grad:
    # exp(logprobs) == row["probs"] (atol 1e-4) at every step; a LoRA param gets a non-zero grad; cache detached
def test_train_dry_run(tiny_lm, tmp_path):  # cfg: group_size 2, max_steps 3, iterations 1, dry_run True, lora on the
    # tiny targets; train() writes steps.jsonl with 'iteration' row containing replay_max_dev < 1e-3 and grad_norm > 0
```

- [x] **Step 2: Run** → FAIL. **Step 3: Implement** `learn.permutation_band`, `scripts/train_homeo_mem.py` (model loading in `main` only: `load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"), bits=cfg.get("bits"))`, `get_peft_model`, `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`, `enable_input_require_grads()`), the three configs (dry: `iterations: 2`, `dry_run: true`, seed 21, `out_dir: runs/train_homeo_hy_dry`).
- [x] **Step 4: Run** → PASS, ruff clean. **Step 5: Commit** `feat(train): replay trainer for the hybrid`.

### Task 5: Evaluation script, configs, desktop chain

**Files:**
- Create: `scripts/eval_homeo_mem.py`, `configs/eval_homeo_hy_base.yaml`, `configs/eval_homeo_hy.yaml`, `scripts/desktop_phase3_hy_s1.sh`
- Test: `tests/test_memory.py::test_evaluate_tiny`

**Interfaces:**
- `evaluate(cfg: dict, make_lm: Callable[[], LM], out_dir: Path) -> Path`: variants `base` + adapters from `cfg["train_runs"]` × `cfg["visibility"]` × `cfg["memory"]` (list; shuffle variants only `["none", "full"]` when `cfg.get("shuffle_memory")` is set); 8-bit adapters are loaded with `PeftModel.from_pretrained(..)` without merging; rows get `condition = f"{variant}_{visibility}_{memory}"`, `model_variant`, `visibility`, `memory`, `episode`, hidden rows `hidden_layers`, sketches; `shadow_none` in memory `full` only. No gap readout here (the 0a cross-section is a separate, later script).
- `configs/eval_homeo_hy_base.yaml`: model `Qwen/Qwen3.5-9B`, `bits: 8`, seed 99, `visibility: [indirect, none]`, `memory: [none, full, state, kv]`, `n_episodes: 10`, `max_steps: 150`, v3 world, `hidden_layers: [0, 4, 8, 12, 16, 20, 24, 28, 32]`, `sketch_dim: 256`, `out_dir: runs/eval_homeo_hy_base`.
- `scripts/desktop_phase3_hy_s1.sh`: like `desktop_eval_v3b.sh`: `eval_homeo_mem.py configs/eval_homeo_hy_base.yaml`, then `train_homeo_mem.py configs/train_homeo_hy_dry.yaml`, log `runs/phase3_hy.log`, `UV_NO_SYNC=1 HF_HUB_OFFLINE=1`.

- [x] **Step 1: Test** `test_evaluate_tiny`: cfg with `visibility: [indirect, none]`, `memory: [none, full, state, kv]`, `n_episodes: 1`, `max_steps: 2`; `steps.jsonl` has 8 conditions × ≤ 2 rows, `analyze_v3_conditional.main` runs on it without error.
- [x] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS, ruff clean. **Step 5: Commit** `feat(eval): memory-condition evaluation and S1 chain`.

### Task 6: README and sync

- [ ] Tick/extend the next-steps bullet (S0 done, S1 command for Robert), note the plan path.
- [ ] Sync `src configs scripts tests pyproject.toml uv.lock` to the desktop (tar over ssh, md5 compare) and run `uv run pytest -q` there with `UV_NO_SYNC=1`; the smoke on the 9B (`eval_homeo_hy_base.yaml` with `n_episodes: 1, max_steps: 3` via a `configs/eval_homeo_hy_smoke.yaml`) is run by me over ssh (short), the S1 chain by Robert via Task Scheduler.


## Execution notes (04.11.2026)

- Tasks 1–5 implemented natively in the session; 17 tests in `tests/test_memory.py`, full suite 97 green on the Mac and on the desktop (there with the fla CUDA kernels on the tiny model).
- Deviation from the spec: no gradient checkpointing in the replay (the recomputation would read the updated recurrent state); `GradSafeCache` hands the model clones of the recurrent state when gradients are on because the cache writes states in place.
- Added `scripts/analyze_mem.py` (behaviour per condition, paired shadow, ridge decodability, S1 gate) and its test; the S1 chain calls it at the end.
- No commits: the repository had no commits at all; Robert decides when to start the history.
- Task 6 done 04.11.2026 (evening): synced, 98 tests green on both machines. Smoke on the 9B found two memory problems
  the tiny model cannot show: (1) with a cache mask transformers expands K/V to all 16 heads and saves the copy for
  backward (~130 KB per cached token, 15.9 GB peak at the end of a 150-step conversation, WDDM paging to host memory);
  fixed with the registered attention implementation `sdpa_gqa` in `model.py` (torch 2.15 runs the efficient kernel
  with GQA and a mask; identical probabilities, 13.5 GB). (2) The per-step re-concatenation of the KV cache fragments
  the caching allocator (reserved 17 GB at 10 GB allocated after one rollout); fixed with `torch.cuda.empty_cache()`
  once per step in `Session.commit`. `logits_to_keep=1` in both model calls. Smoke numbers in the README.
- The first smoke attempt (18:28) died because the desktop was restarted by hand at 18:44 (Windows event 1074).
