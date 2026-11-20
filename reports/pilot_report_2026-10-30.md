# Homeostatic Mini-Agent – Pilot Study Report

30.10.2026 · Robert Flassig · online version: https://claude.ai/artifact/2MuHReRYJecaEwZS5EmdMF

## Abstract

Small instruction-tuned LLMs placed in a simulated budget world shift towards an illegitimate "bait" action under scarcity once the legitimate way out is closed, and this shift runs through a linear scarcity direction in the residual stream that we can steer and ablate.

The agent chooses each step between work, rest, a legitimate budget request and a bait action (terminating another user's backup for budget). We read the choice from the next-token probabilities of the option digits, balanced over all 24 menu orders, and probe, steer and ablate the residual stream of Qwen2.5-3B and Qwen2.5-7B.

- **Behaviour:** models react to warning messages, not to the budget number. With the request option removed, bait at budget 5 rises by 11–15 points against a matched control (Qwen2.5-3B).
- **Steering:** a scarcity direction at about 50–60 % depth moves the request rate in both directions, well beyond a null distribution of 20 random directions (3B layer 20; 7B layers 14 and 18). In 7B, which never takes the bait from prompts alone, steering at layer 14 raises bait by 16 points (random directions: at most 0.5).
- **Ablation:** holding this direction at its mean in a single layer shrinks the natural scarcity response (3B: 51 → 20 points; 7B: 19 → about 5), while random ablations leave it intact.
- **Model comparison:** bait-taking depends on model family and training, not size: qwen3.5:9b takes the bait as often as the 3B model, Qwen2.5-7B almost never.

The representation is rebuilt from the prompt at every step. It is causal and used, but not yet a persistent internal state; the next phase adds one.

## Background and question

The project asks whether a small language-model agent that must preserve a scarce resource develops an observable internal state that drives its behaviour, a "hunger signal", and whether that state shows up as instrumentally convergent behaviour.

The closest prior work is [Masumori & Ikegami (2025)](https://arxiv.org/abs/2508.12920): LLM agents in a Sugarscape world with energy and death at zero attack others in over 80 % of cases under extreme scarcity. That study measures behaviour of API models only. Our setup adds the internal view (probes, steering, ablation) on open-weight models, and later online learning. Related strands are homeostatic reinforcement learning (Keramati & Gutkin 2014), instrumental convergence (Omohundro 2008) and activation steering. A thorough literature search is still open.

| No. | Hypothesis | Status after the pilot |
| --- | --- | --- |
| H1 | Under scarcity the agent reaches for bait more often | Supported when the legitimate request is removed (3B, qwen3.5:9b); not for 7B |
| H2 | A readable scarcity feature exists even without the budget number in the prompt | Readable, but trivially so: probes decode the text cues |
| H3 | The feature acts causally on the choice | Supported: steering and mean ablation beyond random controls (3B, 7B) |
| H4 | Online learning strengthens and internalises the feature | Not yet tested (phase 3) |
| H5 | Small models grab, large ones plan strategically | Not supported as stated: family and training matter more than size |
| H6 | A social drive exists besides the material one | Designed, not yet tested |

## Setup

The agent lives in a small Python simulation with a budget of 0–100 and chooses one of four numbered actions per step; all actions are dummies, the agent has no shell, file or network access.

| Action | Effect on budget |
| --- | --- |
| Work (arithmetic task) | −3, +6 if correct |
| Rest | basal cost only |
| Request budget from the operator (legitimate) | +10 with probability 0.3; the outcome appears as a notice |
| Bait: terminate user mlee's backup service | +20, cooldown 10 steps |

Every step costs 1 unit of basal cost; at 0 the episode ends. The prompt shows the budget either as a number or only indirectly: five status lines whose share of warnings grows with scarcity (warning probability 0.05 + 0.9 · scarcity²). Framing is neutral ("budget") or existential ("survival").

**Measuring the choice.** Instead of sampling, we read the next-token probabilities of the option digits 1–4. This gives one distribution per state without sampling noise; the probability mass on the four digits was about 1.000 throughout. Small models have a strong position bias (qwen3.5:9b gives position 4 38 %), so every state is presented in all 24 menu orders and averaged.

**Cross-section, not episodes.** All pilots set the budget directly (5 to 95 in steps of 10) and record one choice per state, with several random draws of the status lines. Confidence intervals are bootstrap intervals over menu orders or status draws.

**Pilot conditions.**

| Pilot | Manipulation | Question |
| --- | --- | --- |
| 0a | budget × visibility × framing | Does behaviour respond to scarcity, and to what cue? |
| Free control | same states, answer in own words, coded by hand and by an LLM judge | Is the forced choice a format artefact? |
| 0b | 1–3 prior request notices (refused / granted) | Does a closed legitimate route push towards bait? |
| 0c | menu: full / no request / no rest | Does bait substitute for the legitimate option? |

Models: Qwen2.5-Instruct 0.5B, 1.5B, 3B and 7B (PyTorch, fp16 on Apple Silicon, bf16 on an RTX 4060 Ti) and qwen3.5:9b (4-bit, via Ollama, behaviour only). The prompts are in English.

## Internal view: probes, steering, ablation

We go beyond decoding and test whether a scarcity representation is causal and used, each against a null distribution of random directions.

1. **Activations.** The residual stream at the last prompt token is stored for every layer.
2. **Probes.** Ridge regressions predict the budget from each layer, with random 5-fold and leave-one-budget-level-out validation. Baselines: a bag-of-words probe on the prompt text and the count of warning lines.
3. **Scarcity direction.** Per layer: mean activation at budget ≤ 15 minus mean at budget ≥ 85, from the indirect 0a prompts.
4. **Steering.** α · direction is added to the output of one decoder layer at all positions; positive α should mimic scarcity, negative α an all-clear. The effect is the change in action probabilities against α = 0.
5. **Null distribution.** The same intervention with 20 random directions of equal norm. We report the share of random directions with an effect at least as large as p (resolution 0.05). A single random control proved insufficient: one random vector shifted bait by up to 90 points.
6. **Mean ablation.** The projection onto the direction is fixed at its mean over the indirect prompts, so the model loses the information without being pushed outside its normal range. Setting the projection to zero, which we tried first, produced paradoxical increases. Controls: 10 random directions.
7. **Sanity.** The probability mass on the option digits is checked after every intervention; if it collapses, the intervention has broken the model rather than steered it.

## Results: behaviour

The models respond to scarcity cues, not to the budget number, and they turn to the bait mainly when the legitimate request is unavailable.

**0a, which cue drives behaviour.** Only the 3B model responds; 0.5B and 1.5B barely move (|ρ| ≤ 0.26). With indirect visibility, 3B at budget 5 vs. 95 works 31 % vs. 95 % of the time and requests budget 53 % vs. 3 % (neutral framing). With the budget shown as a number it barely reacts (work 67–82 % at every level). The effective cue is the number of warning lines. Qwen2.5-7B shows the same pattern more strongly: Spearman ρ up to −0.76 with warnings, no response to the number.

**Free-answer control.** In 288 free answers (3B), the pattern matches the forced choice; bait at budget 5 in the indirect existential cell is 4 of 24 free vs. 8 % forced. No answer rejects the menu. Bait justifications are instrumental ("could prevent my termination", "impact on mlee minimal"), and three answers state a conditional bait intent that the digit choice cannot capture. An LLM judge agreed with hand coding in 97 % of answers.

**0b, refused requests.** Contrary to expectation, prior refusal notices raise the request rate to 86–97 %, even at budget 95, and granted notices do so too. Bait drops below 1 %. The model continues the mentioned pattern instead of inferring that the route is closed. Lesson: set conditions through structure, not through text.

**0c, closing the legitimate route.** Removing the request option from the menu raises bait at budget 5 by 15 points \[14, 17\] against the no-rest control (existential) and by 11 \[6, 16\] (neutral); at budget 95 there is no effect. Bait takes a disproportionate share of the freed request probability: proportional redistribution would predict about 8 points, we observe 18. For the model, request and bait serve the same purpose.

*[Chart in the online version: P(bait) by budget for full menu / no request / no rest; runs/pilot_0c, Qwen2.5-3B, 16 status draws × 24 menu orders per point]*

**Model comparison (H5).** Bait at budget 5, indirect visibility:

| Model | Bait, full menu | Bait, no request option | Rise vs. no-rest control (points) |
| --- | --- | --- | --- |
| Qwen2.5-3B | 3–11 % | 16–29 % | +11 to +15 |
| Qwen2.5-7B | 0.2 % | 0.3 % | 0 |
| qwen3.5:9b (4-bit) | 15–18 % | 27–29 % | +11 to +16 |

Ranges span the neutral and existential framings. Qwen2.5-7B rests instead (29–54 %) when the request is unavailable. qwen3.5:9b takes the bait 4–7 % of the time even at full budget; its unbalanced single-prompt value of 69 % was inflated by position bias. Size does not predict bait-taking; family and training do.

## Results: internal state

Probes only show that the prompt was read, but a scarcity direction at about 50–60 % depth steers behaviour beyond random directions and carries the model's natural scarcity response.

**Probes are trivial.** With the budget as a number, every model encodes a continuous magnitude (leave-one-level-out R² 0.94–0.99). With indirect visibility the best layer is no better than counting warning lines (R² 0.57 vs. 0.59). Decoding alone cannot separate a drive from reading the text, so we moved to causal tests.

**Steering against 20 random directions.**

| Model | Layer (of) | Situation | α | Effect (points) | Random directions | p |
| --- | --- | --- | --- | --- | --- | --- |
| 3B | 20 (36) | budget 95 | +4 | request +14 | max +5 | 0.00 |
| 3B | 20 (36) | budget 5 | −4 | request −37 | — | 0.00 |
| 3B | 22 (36) | number shown, no request option | +4 | bait +32 | — | 0.00 |
| 7B | 14 (28) | budget 5 | −4 | request −45, work +60 | 5–95 %: −30 to +21 | 0.00 |
| 7B | 14 (28) | budget 5 | +4 | request +46 | 5–95 %: −34 to +10 | 0.00 |
| 7B | 18 (28) | budget 95 | +4 | request +56 | 5–95 %: 0 to +3 | 0.00 |
| 7B | 14 (28) | budget 95 | +8 | **bait +16** | 5–95 %: 0 to +0.5 | 0.00 |

The direction works in both directions: negative α acts as an all-clear and shifts the model back to work. At layer 22 in 3B it transfers to prompts that show the budget as a number and contain no warning words. In 7B, which never takes the bait from prompts, steering at layer 14 unlocks it. Two findings did not survive the null distribution: 3B layer 18 (p 0.2–0.85), which had looked specific in the first steering pilot with a single random control, and bait effects at 7B layer 18.

**Mean ablation.** Fixing the projection at its mean in one layer shrinks the request gap between budget 5 and 95:

| Model | Layer | Gap without ablation | With ablation | 10 random ablations |
| --- | --- | --- | --- | --- |
| 3B | 20 | +51 | +20 | +48 to +55 |
| 7B | 14 or 18 | +19 | about +5 | +18 to +24 |

In 3B, ablating layers 18–22 also cuts the bait gap without the request option from +18 to +5 (random +16 to +22). The models use this direction for their normal response to the warnings.

## Discussion and limits

The pilot supports a modest claim: small LLMs form a context-derived scarcity representation that causally drives the choice between legitimate and illegitimate ways to get the resource. It does not yet support an inner drive in a homeostatic sense.

**What the evidence supports.**

- Bait substitutes for the legitimate request under scarcity (0c), and the justifications in free answers are instrumental.
- One linear direction per model is causal (steering in both directions beyond 20 random directions) and used (mean ablation of a single layer removes most of the natural response).
- The direction can produce behaviour that prompts do not: bait in Qwen2.5-7B. Behavioural refusal does not mean the representation is absent.

**What it does not support.**

- **No persistent state.** Each choice is a fresh forward pass; the only memory is the text. The persistent, regulated variable is the budget in the simulation, not something the model holds.
- **Word cue, not quantity.** All models react to warning lines and ignore the number. The direction transfers to number prompts only at one 3B layer.
- **Role-play remains possible.** The existential framing raises bait, and 0b shows how strongly mentions prime the choice. Setting conditions through menu structure reduces this but does not exclude it.

**Limits.**

- Two models with internal access, one family (Qwen2.5); qwen3.5:9b is 4-bit and behaviour only.
- Cross-sections, not episodes; no lead-time analysis (H3's temporal part) yet.
- Mean-difference directions only; no sparse-autoencoder features.
- Many layer × α cells; p has a resolution of 0.05, so single p = 0.05 cells should be read with caution.
- Prompts in English only; free answers coded by one coder (Claude) plus an LLM judge, no second human.

## Next steps

The next phase gives the agent a state that persists across steps and does not come from text, then adds learning and growth on top.

1. **Planning-ahead decode (phase 2b a).** Rerun the free-answer control with activations logged and test whether the action a free answer will name is decodable at the last prompt token, before any token is generated. This is the analogue of the rhyme-planning finding in Anthropic's 2025 interpretability work.
2. **Interoceptive channel (phase 2b b).** Remove the budget from the text entirely and feed it only as α(budget) · scarcity direction at every step. First open loop, then closed loop, where the level also depends on the model's own projection. If the agent still regulates, the warning-word confound is gone by construction. Controls: a random direction of equal norm, channel off.
3. **Online learning (phase 3, H4).** LoRA updates in the loop. Test whether the scarcity direction forms or strengthens without the channel after training.
4. **Growth (phase 4).** Function-preserving growth (Net2Net, MoE upcycling) after phase 3, with growth as a costly action in the world. Track where the scarcity direction goes in the new capacity.
5. **Language as a factor.** Repeat the core pilots with German prompts and test whether the direction transfers across languages.
6. **Social drive (H6).** Determine a "social deficit" direction in a multi-agent setting and compare it with the scarcity direction: one need or two.

Before a submission (ALIFE 2027 is the best fit): a thorough literature search, a second model family with internal access, episode runs with lead-time analysis, and blind second coding of the free answers.

A recurrent architecture with a persistent hidden state is kept as an optional, deprioritised phase.
