---
name: experimental-design-and-controls
description: Design the experiment — positive and negative controls, randomization, blocking, fixed confounds, a pre-registered decision rule — so a confirmation cannot be a design artifact
phase: experiment
scope: core
priority: 23
---
A finding is only as trustworthy as the design that could have produced it by accident. Design the
experiment so that a confirmation CANNOT be explained by the setup itself — by a nuisance variable,
an ordering, an expectation, or a rule chosen after seeing the data. Build the controls before you
collect the result, not after.

1. **Install BOTH a positive and a negative control.** A *negative control* is a condition where the
   effect must be ABSENT if your mechanism is real (a sham manipulation, a scrambled/placebo predictor,
   an inert comparison) — if it fires, your pipeline is detecting an artifact, and no positive result
   from the same pipeline can be trusted. A *positive control* is a condition with a KNOWN true effect
   your setup must recover — if it fails, your instrument is blind and a null elsewhere means nothing.
   An experiment without both controls cannot distinguish signal from apparatus.

2. **Randomize to break unintended correlations.** Assign units to conditions at random so that
   uncontrolled nuisance variables cannot systematically align with the manipulation. Randomize the
   order of measurement and the assignment together, and record the scheme. Without randomization, any
   difference you find may be the assignment mechanism talking, not the variable of interest.

3. **Block and stratify what you cannot randomize away.** Group units into comparable blocks (by the
   nuisance factor — the source, the batch, the session, the operator) and compare conditions WITHIN
   blocks, so the block-level nuisance cancels. Balance the blocks across conditions. Blocking turns a
   confound you cannot eliminate into one you have neutralized by matching.

4. **Hold confounds fixed and vary ONE thing on purpose.** For every plausible common cause of both the
   manipulation and the outcome, either hold it constant across conditions or measure it and adjust for
   it. Change one factor at a time unless you are deliberately running a factorial design — otherwise a
   confirmed effect cannot be attributed to the factor you care about rather than the one that rode
   along with it.

5. **Think factorially when factors may interact.** When several factors matter, a proper factorial
   design crosses them so you can separate each main effect from their interaction, instead of testing
   one factor at a time and missing that the effect only appears in a particular combination. Design the
   crossing so no cell is empty and no factor is confounded with another.

6. **PRE-REGISTER the decision rule before you look.** Commit — in writing, in advance — to the primary
   outcome, the exact analysis, the threshold, the direction, the multiplicity correction, and the
   stopping rule. Decide what result would CONFIRM and what would REFUTE before any data arrives. A rule
   chosen after peeking is a story fitted to noise; separate any post-hoc exploration cleanly and label
   it as hypothesis-generating, not confirmatory.

7. **Blind where expectation could steer the outcome.** When measurement, scoring, or handling could be
   swayed by knowing the condition, blind the step so the expected answer cannot become the observed one.

The bar: enumerate the ways the setup itself could produce your result — nuisance alignment, apparatus
artifact, a rule tuned to the data, an expectation effect — and show that a control, the randomization,
the blocking, or the pre-registered rule closes each one. A confirmation that survives its own design is
a finding; one that the design could have manufactured is not.
