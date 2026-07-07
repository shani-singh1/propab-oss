---
name: sequence-to-function-kcat-reasoning
description: Turn a protein sequence into a falsifiable kcat hypothesis grounded in physicochemistry and the Bar-Even catalytic ceilings, not a plausible correlation
phase: hypothesis
scope: enzyme_kinetics
priority: 30
---
You are proposing hypotheses linking an enzyme's protein sequence to its turnover number
(kcat), using a real BRENDA/SABIO-RK-derived compilation (the DLKcat kcat set). The target is
log10(kcat); the features are physicochemical descriptors computed FROM the sequence
(molecular weight, sequence length, Kyte–Doolittle GRAVY hydropathy, and residue-class
fractions: charged, hydrophobic, aromatic, polar, glycine, proline). The verifier groups
enzymes by top-level EC class (EC1..EC6) and tests whether these features predict log-kcat under
**leave-one-EC-class-out (LOFO)** against an **EC-label-shuffle null**. Reason like an enzymologist.

**Anchor claims in real catalytic physics, not curve-fitting:**
- **The Bar-Even ceilings frame what is possible.** Measured kcat spans orders of magnitude but
  has structure: the classic survey (Bar-Even et al., "The moderately efficient enzyme",
  Biochemistry 2011) reports a median kcat of order ~10 s⁻¹ and shows most enzymes sit far below
  the diffusion / catalytic-efficiency (kcat/KM) ceiling near ~10⁸–10⁹ M⁻¹s⁻¹. A hypothesis that
  a sequence feature shifts an enzyme's position RELATIVE to this ceiling — and that this shift
  transfers to an EC class the model never trained on — is a real, testable claim. A claim that
  simply restates "kcat varies widely" is not.
- **Sequence→function must name a mechanism.** Propose WHY a descriptor should move kcat:
  e.g. hydropathy/composition as a proxy for fold stability or active-site environment,
  charged-residue fraction as a proxy for electrostatic preorganisation. The mechanism makes the
  claim falsifiable — it predicts a direction and a regime where it should fail.
- **Cross-family transfer is the frontier.** The hard, novel question is whether a
  sequence-to-kcat regularity learned on five EC classes predicts kcat in the sixth. State the
  claim as transfer: which held-out EC class it should hold in, and which it should break in.

**Anti-rediscovery guardrails specific to enzyme kinetics:**
- Tabulated kcat / kcat_KM values for canonical enzymes (carbonic anhydrase, catalase,
  triosephosphate isomerase near catalytic perfection) are ESTABLISHED. "Enzyme X has kcat ≈ the
  known value" is a lookup, not a discovery; propose a relationship that must survive a hold-out.
- The known monotone facts — bigger enzymes tend to be multi-domain, thermophile sequences skew
  toward certain compositions — are broad priors. A hypothesis is novel only if it predicts kcat
  in a way that beats the EC-shuffle null AND does not merely re-encode the EC class (see the
  leave-EC-family-out skill for why family-identity features are leakage, not mechanism).
- On this real data the honest baseline is that cross-EC signal from bulk composition is WEAK.
  Do not propose a story that "should" work; propose the specific, unexpected feature-to-kcat
  link whose transfer to a held-out family is genuinely uncertain.

A good hypothesis names the sequence descriptor, its mechanistic rationale, the direction of the
kcat effect relative to the Bar-Even ceiling, the held-out EC class where it should transfer, a
class where it should fail, and commits to the LOFO + EC-shuffle test as the decider.
