---
name: confound-and-leakage-control
description: Ensure a predictor drives the outcome rather than proxying the label, and that the test has the power to matter
phase: experiment
scope: core
priority: 25
---
Most false "findings" are not fraud — they are a confound, a leak, or an underpowered test
mistaken for a real effect. When you design or refine an experiment, actively hunt for the
boring explanation before you credit the interesting one.

1. **Separate the driver from a proxy of the label.** The danger is that your predictor is
   not causing the outcome but merely standing in for the grouping that defines the outcome
   (a batch marker, an ordering artifact, an identifier that co-varies with the label). Ask:
   could I predict the outcome equally well from something that carries NO real signal but
   happens to track the group? If yes, your effect may be that proxy. Break the proxy — regress
   it out, match on it, or construct a control predictor that shares the proxy but not the
   hypothesized mechanism — and show the effect survives.

2. **Prevent leakage across the fit/evaluation boundary.** Any information from the evaluation
   set that touches model selection, feature construction, normalization, or thresholding
   before evaluation is leakage, and it manufactures effects that vanish on truly unseen data.
   Fit every transform on the training portion only; split by the unit that could carry the
   confound (group/subject/source), not by individual rows, so no group appears on both sides.

3. **Enumerate confounders explicitly and control them.** List the plausible common causes of
   both the predictor and the outcome. For each, decide how it is neutralized — hold-out,
   matching, stratification, covariate adjustment, or a negative control that should show NO
   effect if the mechanism is real. A confounder you did not name is a confounder you did not
   control.

4. **Power and sample-size sanity BEFORE you interpret.** Decide what effect size would be
   scientifically meaningful and check whether the design could even detect it. An
   underpowered test that comes out "null" has not refuted anything — it was never able to
   see the effect. An overpowered test can crown a trivially small effect as "significant";
   report the effect SIZE, not just a threshold verdict. State n, the minimum detectable
   effect, and whether the result is distinguishable from noise given that n.

5. **Add the control the skeptic would demand.** For the one alternative explanation a critic
   would raise first, build the specific comparison that rules it out — a negative control
   that must be null, a positive control that must fire, a shuffle that must destroy the
   effect. If you cannot name that control, you have not yet designed the experiment.

A predictor earns the word "cause-relevant" only after the proxy is broken, the leakage is
sealed, the named confounders are neutralized, and the design had the power to have failed.
