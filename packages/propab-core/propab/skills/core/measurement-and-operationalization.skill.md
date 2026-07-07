---
name: measurement-and-operationalization
description: Turn an abstract construct into a defensible measured quantity — and never mistake a finding about a proxy for a finding about the construct
phase: hypothesis
scope: core
priority: 16
---
Every claim rests on a chain: an abstract construct you care about → an operational definition → a
concrete measured quantity. The claim can only be as strong as its weakest link, and the most common
silent failure is that the measured quantity stands in for something OTHER than the construct — most
dangerously, for the group label itself. Pin the chain down before you design the test.

1. **Name the construct, then its operational definition.** State the abstract thing the hypothesis is
   really about, then state the exact, repeatable procedure that turns it into a number: what is
   recorded, on what unit, with what instrument, under what conditions. If two people following your
   definition would produce different quantities, the definition is not yet operational — sharpen it.

2. **Argue construct validity — does the quantity capture the concept?** A measurement is valid only to
   the extent the number moves with the construct and not with irrelevancies. Ask: does it cover the
   whole construct or only one facet (under-representation)? Does it also capture things the construct
   excludes (contamination)? A quantity that correlates with the construct in convenient cases can still
   diverge in exactly the regime you want to claim — say where you expect validity to hold and where it
   should break.

3. **Characterize measurement error and its DIRECTION.** No quantity is exact. State whether the error is
   random (noise that widens uncertainty and, if it afflicts the predictor, generally biases an estimated
   effect toward zero) or systematic (a bias that shifts every reading the same way and can create or
   erase an apparent effect). Direction matters: random error mislabeled as "just noise" can hide a real
   effect, while a systematic error aligned with the grouping can fabricate one. Say which kind you have
   and which way it pushes the conclusion.

4. **Hunt the PROXY trap.** A proxy is a measured variable used in place of the construct because the
   construct is hard to observe directly. It is a trap when the proxy tracks something else that co-varies
   with the outcome. The worst case: the quantity is really standing in for the GROUP LABEL — an
   identifier, batch marker, collection order, or recording artifact that separates the groups for reasons
   having nothing to do with the construct. Test it directly: could a variable carrying NO construct
   signal, but tracking the same grouping, reproduce the result? If yes, you are measuring the label, not
   the concept. Break the proxy — hold the label-correlated nuisance fixed, regress it out, or build a
   control quantity that shares the nuisance but not the construct — and show the effect persists.

5. **Bound the claim to the operationalization.** A result obtained with one operational definition is a
   result ABOUT that operationalization until shown otherwise. State explicitly: "this supports a claim
   about [construct] only insofar as [quantity] validly measures it; under [named alternative
   operationalization] the claim is untested." A finding about the proxy is not a finding about the
   construct — and reporting it as the latter is the error this skill exists to prevent.

Do this at hypothesis time, not after: if you cannot yet name a quantity that measures the construct
rather than the label, you do not yet have a testable hypothesis — you have a construct in search of a
measurement.
