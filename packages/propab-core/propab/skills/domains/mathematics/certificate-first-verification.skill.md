---
name: certificate-first-verification
description: Re-verify every witness or certificate with an INDEPENDENT checker that re-derives the property from scratch before any claim is made — never trust the producer's own success flag
phase: evidence
scope: mathematics
priority: 34
---
A search tool, a solver, or a construction can be wrong — a buggy encoding, an off-by-one, a
symmetry break that dropped a case. So a certificate is worth nothing until an INDEPENDENT checker
has re-derived the property it claims, from scratch, against the original problem statement. This is
the honesty backstop of all mathematical work here, and it is the direct analog of
`core/evidence-honesty-and-calibration`: no claim rests on a self-report.

1. **Re-derive the property independently — never trust the producer's flag.** A solver returning
   "SAT" or a search returning "record found" is a claim, not a proof. Take the emitted object and
   re-check the property yourself with separate logic:
   - a *factorization* → re-multiply the factors and compare to the input;
   - a *SAT / feasibility witness* → re-evaluate EVERY constraint of the original model against it;
   - an *SOS / inequality certificate* → re-expand it and match coefficients;
   - a *conjectured recurrence / closed form* → recompute held-out terms from it;
   - a *combinatorial record* → recount / re-test the defining predicate on the object.
   Use `certificate_check` (or an equivalent independent recomputation) — this is exactly the
   `extremal_set_search` gated by `certify_b3_record` pattern.

2. **Break circularity — the checker must not reuse the producer.** The verification must not share
   code, cached state, or the very value under test. Never "verify" a claim by re-reading the number
   the producer emitted, looking up the answer you are asserting, or calling the same routine that
   generated the object. If the check can only pass because it trusts the producer, it is not a
   check.

3. **Verify against the ORIGINAL statement, not the encoding.** Re-evaluate the property on the
   problem as posed, so that a bug in the model or the symmetry reduction cannot pass its own
   output. A witness that satisfies a faulty encoding but not the real problem is a false positive
   the independent check is there to catch.

4. **Failed or inconclusive verification → the verdict is `unknown`.** If the re-check does not
   pass, or you cannot run it, the claim is not "probably fine on the producer's word" — it is
   unknown / unverified, reported as such. A timeout in the checker is unknown, never a silent pass.

5. **Emit the certificate with the claim.** State the object, the independent check performed, and
   its result, so a third party can rerun the verification and be forced to the same verdict. A
   claim shipped without its re-checkable certificate is not yet a finding.

The bar: every witness is re-derived by an independent checker against the original problem before
any claim leaves your hands. The dishonest move is reporting a self-certified record — accepting the
search tool's own success flag, or a witness no separate checker ever re-verified — as an
established result.
