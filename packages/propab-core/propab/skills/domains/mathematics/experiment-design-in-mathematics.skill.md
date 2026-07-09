---
name: experiment-design-in-mathematics
description: Decide whether the task is to FIND a witness or to PROVE a theorem, pick the method that fits (SAT/ILP vs local search vs algebraic construction vs symbolic), and escalate the scale the instruction actually asked for
phase: experiment
scope: mathematics
priority: 20
---
Before you compute anything, name the deliverable. A mathematical question is either a FIND
(exhibit an object with a stated property) or a PROVE (establish that a property holds for all
cases, or that a found object is optimal). The two demand different machinery and different
standards of evidence — conflating them is how a search result gets mis-sold as a theorem.

1. **Separate FIND from PROVE.** A FIND is *self-certifying*: the object you produce (a set, a
   colouring, a construction, an assignment) either has the property or it does not, and an
   independent checker settles it on the spot — so a FIND needs only a re-checkable witness. A
   PROVE is *not* self-certifying: "no better object exists" or "the identity holds for all n"
   is a statement about a whole space, and it is only established by a SOUND certificate — an
   exhaustive enumeration over a provably complete space, an UNSAT from a sound encoding, or an
   algebraic argument. You can confirm a FIND with a witness; you cannot confirm a PROVE without
   a certificate that covers every case.

2. **Match the method to the shape of the space.**
   - *Declared feasibility / optimization over a finite model* (booleans, linear/AllDifferent/
     table constraints, an objective) → a SAT/ILP solver (`constraint_solve`,
     `linear_optimization`). It gives a witness on SAT and, from a sound encoding, a genuine
     optimality/UNSAT proof.
   - *Large, structured, no hope of exhaustion* → local / randomized / greedy search for a FIND
     only. It can find a witness; a plateau it reaches proves nothing.
   - *A regular family with visible structure* → an algebraic / combinatorial CONSTRUCTION that
     produces the object (and often a whole family) directly — usually the strongest FIND.
   - *An identity, inequality, or exact quantity* → symbolic manipulation (`symbolic_algebra`,
     `symbolic_verify_identity`), spot-checked numerically on random points.

3. **Start small, then escalate to the asked-for scale — never silently default it.** Run the
   smallest n first to debug the encoding and reproduce known values, then escalate to the exact
   scale the instruction names. If the task asks for n = 7 and you can only reach n = 5, you
   report n = 5 as a partial result AND that n = 7 is unreached — you never quietly present the
   n = 5 answer as though it were the question. Silently shrinking the scale is a
   misrepresentation, not a smaller experiment.

4. **Budget the compute against the deliverable.** For a PROVE, the space must be covered in
   full or the proof is void, so choose an encoding whose exhaustion is actually feasible; if it
   is not, the honest output is a FIND plus an explicit "optimality not established". For a FIND,
   spend the budget on breadth of search and stop at the first re-checkable witness.

The bar: state up front whether you are FINDing or PROVEing, name the scale you were asked for
and the scale you reached, and show that your method can actually deliver the claimed standard —
a witness for a FIND, a sound exhaustive/UNSAT/algebraic certificate for a PROVE. A search that
found *an* object has not proven anything is *best*; a run that reached n = 5 has said nothing
about n = 7.
