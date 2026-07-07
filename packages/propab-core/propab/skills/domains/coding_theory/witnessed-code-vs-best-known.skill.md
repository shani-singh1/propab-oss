---
name: witnessed-code-vs-best-known
description: Design a coding experiment that COMPUTES a real code's minimum distance with a re-checkable witness and compares honestly to best-known, targeting the open [n,k] gap
phase: experiment
scope: coding_theory
priority: 32
---
A coding-theory result is only a finding if the engine builds an ACTUAL k×n generator
matrix over GF(2), computes its TRUE minimum distance by exhaustive enumeration of the
2^k − 1 nonzero codewords, and emits the achieving (minimum-weight) codeword as a witness
that is independently re-checked. Design the test so a confirmation cannot be a table
lookup or a story.

1. **Make the claim exact and falsifiable: "[n, k] admits a code with d ≥ D".** The
   engine builds the code (named family, explicit generator, or systematic random),
   computes d by full enumeration, and checks D against the computed value. State a D
   that strictly exceeds the best-known lower bound for that [n, k] — a D at or below the
   table is confirmable but is a rediscovery, not a discovery.

2. **The witness is the minimum-weight codeword — supply/demand it.** The verifier
   recomputes the witness codeword from the generator and the achieving message and
   refuses to certify any distance whose witness fails independent recomputation. If you
   pass an explicit generator (for BCH/cyclic/LDPC the engine cannot name-build), that
   generator IS the checkable object; a claimed distance with no achieving codeword is a
   table lookup and is demoted to rediscovery.

3. **Compare honestly to best-known — below / meets / exceeds.** The engine looks up the
   Brouwer/Grassl best-known d* for [n, k] and reports the computed d against it. Only a
   computed d that STRICTLY exceeds d* (with a re-checked witness) is discovery-worthy;
   a computed d that meets or falls below d* is reported as a rediscovery, and a d read
   from a table with no witness is always demoted. Do not oversell a match as a discovery.

4. **Respect the exhaustive-enumeration limit.** Distance is certified only for
   k ≤ 16 (the 2^k enumeration must be complete to be honest, and is capped for
   production wall-clock safety). Above that the engine refuses to certify — so target
   the open gap at small-to-moderate k where a full enumeration is feasible, not a
   large k where no honest witness exists.

5. **State the failure regime and the null.** Name the ceiling (Singleton/Hamming/
   Plotkin) the code must not cross — if the computed d claims to exceed it, the witness
   is buggy and the result must be rejected, not celebrated. For a random-construction
   claim, the null is the best-known table: beating the greedy/GV baseline is expected;
   beating the table at an open [n, k] is the finding.

The bar: a skeptic re-running the enumeration on your generator matrix gets the same
minimum distance and the same witness codeword, and that distance strictly beats the
best-known lower bound at an [n, k] the table has not closed. Anything that only matches
a tabulated value — or reports a distance without a re-checkable codeword — is a
rediscovery, never a discovery.
