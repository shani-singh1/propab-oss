---
name: causal-inference-and-identification
description: State exactly which causal claim a design can and cannot support — correlation is not causation until an identification strategy breaks the confound
phase: evidence
scope: core
priority: 28
---
An association between a predictor and an outcome is compatible with many worlds: the predictor
causes the outcome, the outcome causes the predictor, a third factor causes both, or the link is
an artifact of how cases were selected. A causal claim is licensed only when the DESIGN rules out
the rival worlds — not when the story sounds mechanistic. Before you attach a causal verb ("drives",
"increases", "because"), work through identification explicitly.

1. **Write the causal question as a would-be intervention.** State it as: "if I set the predictor to
   value X versus Y, holding nothing else fixed by hand, the outcome would differ by Δ." If you cannot
   phrase the claim as an intervention on a manipulable quantity, it is descriptive, not causal — stop
   upgrading it.

2. **Draw the causal map and enumerate confounders.** Sketch the assumed arrows among predictor,
   outcome, and every plausible third factor. A confounder is any common cause with an arrow into BOTH
   the predictor and the outcome; it manufactures association with no causal path. List them by name —
   an unnamed confounder is an uncontrolled one. Also mark any variable on the causal path you must NOT
   adjust for (a mediator) and any common effect you must NOT condition on (conditioning on it opens a
   spurious path). "Control for everything" is wrong; the map tells you what to adjust and what to leave
   alone.

3. **Check temporality and reverse causation.** A cause must precede its effect. Verify the predictor
   was fixed before the outcome was measured; if they were observed together, you cannot exclude the
   outcome driving the predictor. State explicitly whether the design can order cause and effect in time,
   and if it cannot, treat reverse causation as a live alternative that survives.

4. **Name the identification strategy that breaks the confound.** Association becomes causation only
   through a design that severs the common-cause arrow. Pick the one your setting actually supports:
   - **Randomized manipulation** — assignment to the predictor is set by a mechanism independent of every
     confounder, so groups differ only in the predictor. The strongest identification when feasible.
   - **Natural experiment** — an external, as-if-random event shifts the predictor for reasons unrelated
     to the outcome; you exploit that exogenous variation rather than the endogenous kind.
   - **Instrument** — a variable that moves the predictor but touches the outcome ONLY through it and is
     unrelated to the confounders; identification rests on that exclusion assumption, which you must state
     and defend, not assume.
   - **Leave-one-group-out hold-out that breaks the confound** — partition so the suspected common cause
     cannot be exploited (hold out an entire group/source/batch), and show the effect survives on data
     where the confounding structure is absent.
   For whichever you choose, state the assumption it rests on and what would violate it.

5. **State exactly what the design can and cannot support.** Close with a one-line verdict of the form:
   "This design supports a [causal / associational-only] claim about [predictor→outcome] under
   [assumption]; it cannot rule out [named rival: reverse causation / confounder Z / selection]." A
   correlation reported as a correlation is honest; a correlation dressed as a mechanism is the failure
   this skill exists to prevent.

Run this before claiming any effect is causal, including your own: if no identification strategy in step 4
applies to your setup, the correct claim is associational, and you say so plainly.
