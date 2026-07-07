---
name: descriptor-design-for-structure-property
description: Design physically grounded descriptors for the structure->dielectric relationship on Matbench dielectric crystals
phase: hypothesis
scope: materials
priority: 30
---
You are proposing which STRUCTURAL/compositional descriptor predicts the dielectric
response (refractive index target) on the real Matbench `matbench_dielectric` set
(~4,764 pymatgen crystals). The verifier computes descriptors per crystal, groups by
crystal system (7 systems), and tests generalization by leave-one-crystal-system-out
(LOFO). The exposed descriptors are exactly:

`n_sites`, `n_elements`, `mean_Z`, `std_Z`, `std_principal_quantum_n`,
`mean_atomic_mass`, `mass_density`, `mean_electronegativity`, `mean_ionicity`
(mean |χ_i − χ_j| over element pairs), `mean_coordination`, `mp_bandgap`,
`space_group_number`.

Anchor descriptor choices on real dielectric physics, not arbitrary column-mixing:

- **Clausius–Mossotti / polarizability route.** The dielectric constant rises with
  polarizability density: heavier, more polarizable atoms packed densely. Hypothesize
  that `mean_atomic_mass` + `mass_density` (and low `mean_ionicity`) predict higher
  refractive index — a mechanism, not a fit. Predict where it should hold and where it
  should break.
- **Penn-model / band-gap route.** Electronic dielectric response scales inversely with
  the band gap (ε∞ ≈ 1 + (ℏω_p/E_g)²). A claim that `mp_bandgap` NEGATIVELY predicts the
  dielectric target — and does so ACROSS crystal systems — is physically motivated and
  falsifiable by the LOFO.
- **Ionicity / bonding character.** `mean_ionicity` and `mean_electronegativity` capture
  ionic vs covalent bonding, which shifts static vs electronic dielectric contributions.
  A sign-and-mechanism claim here is testable.
- **Coordination / density packing.** `mean_coordination` and `mass_density` proxy how
  tightly the lattice packs polarizable units; hypothesize their role with a direction.

Design principles:
- **Small, mechanistic descriptor sets beat kitchen-sink dumps.** LOFO rewards a
  descriptor that captures real structure→property physics and generalizes to an unseen
  crystal system; a large collinear bag overfits and inflates the in-sample fit without
  surviving LOFO. Prefer 2–4 physically justified descriptors and predict the sign of
  each.
- **State the transfer claim.** Name the crystal system you expect to be HARDEST
  (the held-out one that would refute), e.g. "holds on cubic/tetragonal but the
  polarizability law weakens on triclinic low-symmetry crystals".
- **Read the leakage guardrail before choosing `space_group_number`** — it encodes
  crystal-system identity and is a leakage trap, not a structure→property descriptor.

A good materials hypothesis names the descriptor, the sign, the physical mechanism, and
the held-out crystal system whose LOFO replication would decide it.
