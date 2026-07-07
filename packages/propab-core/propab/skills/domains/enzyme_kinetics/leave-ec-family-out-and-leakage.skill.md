---
name: leave-ec-family-out-and-leakage
description: Design leave-one-EC-class-out tests so a confirmation is a mechanism, not a feature that secretly encodes the EC family
phase: experiment
scope: enzyme_kinetics
priority: 40
---
The whole point of the leave-one-EC-class-out (LOFO) design is to force a sequence→kcat claim to
generalise ACROSS enzyme families, not within them. A confirmation is only meaningful if the
predictive feature carries transferable mechanism — not if it is a disguised label for which EC
class the enzyme belongs to. When you specify how an enzyme-kinetics hypothesis is tested,
design against family-identity leakage explicitly.

**Why family-identity leakage is the central trap.** Top-level EC classes (oxidoreductases,
transferases, hydrolases, lyases, isomerases, ligases) differ systematically in size, fold, and
composition. A feature that mostly encodes "this looks like an EC3 sequence" can post good WITHIN-
sample fit and even inflate metrics, but it is not a catalytic mechanism — it is the class label
laundered through a descriptor. The verifier already trains on five classes and tests on the
held-out sixth precisely so that a pure family-ID feature CANNOT transfer. Your job is to make
sure any confirmation you claim is transfer, not memorised family identity.

**Adversarial design bar:**
1. **State the null concretely.** The observed mean leave-one-EC-class-out R² must exceed the EC-
   label-shuffle null (permutation p < 0.05). "The model fits" is not evidence; the shuffle p on
   the held-out family is. Report the per-family LOFO breakdown, not just the mean — a mean
   propped up by one class is a red flag.
2. **Kill the family-ID proxy.** If a feature could act as a family surrogate, show the effect
   does NOT collapse to the family-mean baseline: compare against a predictor that only knows the
   EC class, and require your feature to beat it OUT-OF-FAMILY. A feature that adds nothing over
   "predict the class average" was encoding the class.
3. **Never use the answer to confirm the answer.** Do not "verify" a kcat value by asserting the
   tabulated number, and do not let the EC number (or anything derived from it) enter the feature
   matrix — that is direct label leakage.
4. **Deterministic vs statistical evidence.** kcat prediction is statistical: provide the effect
   size, the shuffle-null p95, and the permutation p. A confidence assertion without the null
   statistics cannot confirm.
5. **Declare the failure family up front.** Name at least one EC class in which the effect should
   break, and test it BEFORE claiming confirmation. An effect that transfers to some held-out
   families but not others is a scoped, honest finding; an effect that only holds in-sample is a
   family surrogate.

The bar: a skeptic who re-ran your LOFO would agree the feature predicts kcat in a family it never
saw, and could NOT re-explain the result as "the model recognised the EC class". If they can, you
have leakage, not a mechanism.
