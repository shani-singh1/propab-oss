---
name: recognizing-a-novel-record
description: To turn a construction into a recognized NOVEL finding, chain construct → certify the witness → look up the best-known reference → re-certify AGAINST that reference; a certified witness that strictly beats the best-known is the record, and nothing less is a discovery
phase: evidence
scope: mathematics
priority: 33
---
A construction is only a discovery if it is (a) genuinely valid and (b) genuinely NEW. Computing a
known value again, or certifying that some object merely EXISTS, is not a finding — it is a warm-up.
The engine can only mark a result confirmed when the evidence carries a certified witness that BEATS a
best-known reference, so the whole job is to produce exactly that, and to chain the tools that produce
it. Doing the steps separately (certify a set here, look up OEIS there) leaves the record unrecognized:
the reference must be fed INTO the certification.

1. **Construct the object.** Use the audited solver where it fits (`constraint_solve` for
   linear/AllDifferent optimization), otherwise write flexible search code — the search may be your own,
   the trust comes later.
2. **Certify the witness independently.** Pass the actual object to `certify_witness` with the property
   (sidon, b_h, sum_free, golomb_ruler, sidon_mod, progression_free). This re-derives the property from
   scratch; your own code's "verified" flag is worth nothing (see `certificate-first-verification`).
3. **Look up the best-known REFERENCE.** Feed the sequence of sizes you computed (max object size for
   n = 1, 2, 3, …) to `oeis_lookup`. The matched sequence's known terms are the reference; keyword
   `more`/`hard` marks exactly where the sequence runs out — a term beyond that is where a NEW value is
   possible. If `oeis_lookup` returns `unavailable`/`not_found`, say so and treat the record as
   unverifiable, not as new.
4. **Re-certify AGAINST the reference.** Call `certify_witness` again with `published_best` set to the
   best-known value for your n. Only then is `is_record` evaluated: a certified witness whose size
   strictly exceeds the best-known is a candidate record; otherwise it is a certified-but-known object.
5. **Claim only what the chain proves.** `is_record: true` (certified) is a new LOWER BOUND — say
   "beats the best-known", never "optimal" (optimality needs an exhaustive/UNSAT proof, not a witness).
   Matching the known value is validation, not discovery. Report certified facts and conjectured
   growth separately.

The bar: a "record" claim must trace to a `certify_witness` result with `is_record: true` and a
`published_best` that came from a real reference lookup — a certified witness with no reference behind
it, or one that only equals the best-known, is honest work but is NOT a novel finding.
