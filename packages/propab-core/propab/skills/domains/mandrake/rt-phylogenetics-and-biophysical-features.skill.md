---
name: rt-phylogenetics-and-biophysical-features
description: Frame reverse-transcriptase activity hypotheses from biophysical features that predict across RT families, not features that restate known RT phylogeny
phase: hypothesis
scope: mandrake
priority: 30
---
You are proposing hypotheses about reverse-transcriptase (RT) activity across a small panel of
retroelement RT sequences (~56 sequences, 7 evolutionary rt_family groups with ≥4 each). The
target is prime-editing / RT activity (pe_efficiency_pct). The features are handcrafted
biophysical descriptors: a thermal-stability T-series (t40..t80), catalytic-triad geometry
(D1–D2 / D2–D3 distances, triad RMSD, Ramachandran, YxDD strand context), electrostatics (net
charge, isoelectric point, salt bridges, pocket H-bonds), fold similarity (Foldseek TM/LDDT to
HIV1, MMLV, Retron, Group II intron, telomerase, LTR retrotransposon), surface properties
(CamSol, SASA, hydrophobic fraction), and structural motifs. The verifier tests whether a
feature set predicts activity under **leave-one-family-out (LOFO)** against a **family-label-
shuffle null**, alongside a within-family fit, a family-mean baseline, and a permutation p.

**Reason about RT as a phylogenetically structured protein family:**
- **RT families are an evolutionary tree, and the tree is a confound.** The seven rt_family
  groups are clades; sequences within a clade are related, and their features co-vary with
  ancestry. The scientifically interesting claim is a biophysical driver of activity that holds
  when an ENTIRE family is held out — i.e. that transfers across the tree — not one that merely
  tracks which clade a sequence is in.
- **Biophysical mechanism over fold-similarity shortcuts.** Foldseek TM scores to named
  references (HIV1, MMLV, ...) and catalytic-triad geometry are powerful but dangerous: they can
  encode family identity directly (see the cross-clade holdout skill). Prefer a mechanistic axis
  (thermal stability, electrostatic preorganisation of the active site) whose effect on activity
  you can state as a direction and that has a reason to transfer across clades.
- **Discriminate rival mechanisms, don't just verify one.** The strongest hypotheses here pit two
  explanations against each other so a single experiment moves them in OPPOSITE directions —
  e.g. "unified biophysical mechanism (thermal axis survives LOFO) vs family-specific mechanism
  (signal is real only within a clade and collapses across clades)". A test that would refine one
  belief without discriminating between rivals is weaker than one that separates them.

**Anti-rediscovery guardrails specific to RT:**
- The RT phylogeny is ESTABLISHED: the Pfam RVT clan (CL0027; families PF00078/PF07727/PF13456),
  the seven conserved RT sequence motifs, the ~240-aa catalytic core, and the characteristically
  LOW (<~25%) cross-class RT sequence identity are all tabulated (Xiong & Eickbush 1990; Pfam
  2021). A hypothesis that a feature "distinguishes the RT families" or "recovers the RT clades"
  re-derives known phylogeny — that is a rediscovery, not a discovery.
- A within-family signal that is really the conserved-motif or clade structure is a rediscovery;
  a cross-family biophysical driver of activity is a discovery. Aim for the latter.
- Prefer claims where the OUTCOME is genuinely uncertain: on this panel the honest prior is that
  many features track family identity and collapse under LOFO. Target the specific, surprising
  biophysical axis whose cross-family transfer nobody has established.

A good mandrake hypothesis names the biophysical axis, its mechanistic link to RT activity, the
held-out family where it should transfer, a family where it should break, states whether it
predicts survival or collapse under LOFO, and commits to the LOFO + family-shuffle test (with the
family-mean baseline as the thing it must beat) as the decider.
