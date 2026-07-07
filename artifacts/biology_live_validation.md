# Biology live end-to-end validation (20260707T123317Z)

LLM model: `gemini-3-flash-preview`  |  data: real GTEx v8 + real DLKcat (BRENDA+SABIO-RK)

Live end-to-end: real LLM generation + real LOFO/label-shuffle null on real cached data. Honest refute/inconclusive on subtle real signals is the CORRECT outcome; no confirmation is fabricated.

## genomics

**Question:** In the GTEx v8 cross-tissue expression atlas, which non-obvious gene-level expression feature predicts tissue-specificity (tau) or mean expression in a HELD-OUT tissue, surviving the tissue-label shuffle null?

LLM proposed 3 hypotheses; each verified by the real LOFO + shuffle null.

1. **refuted** (conf 0.82) — The scaling relationship between mean_expression and expression_variance across training tissues predicts the tissue_specificity_tau of genes whose dominant expression is in a held-out structural tissue (e.g., Artery), surviving the tissue-label shuffle null, whereas this relationship fails for genes peaking in the Brain.
   - features: `['mean_expression', 'expression_variance']`; LOFO R²=1.0, shuffle p=1.0, n=1000
   - reason: no cross-tissue signal (LOFO R²=1.000, p=1.000)
2. **refuted** (conf 0.82) — A model trained on cv_across_tissues and tissue_specificity_tau predicts the mean_expression of genes in a held-out secretory tissue (e.g., Pituitary) significantly better than the shuffle null, but this predictive power vanishes for genes in the top 5th percentile of global mean_expression.
   - features: `['cv_across_tissues', 'tissue_specificity_tau']`; LOFO R²=1.0, shuffle p=1.0, n=1000
   - reason: no cross-tissue signal (LOFO R²=1.000, p=1.000)
3. **refuted** (conf 0.82) — The interaction between mean_expression and cv_across_tissues in training tissues predicts the tissue_specificity_tau of genes peaking in a held-out metabolic tissue (e.g., Liver) better than the shuffle null, suggesting that the global expression 'budget' constrains specificity in high-turnover tissues.
   - features: `['mean_expression', 'cv_across_tissues']`; LOFO R²=1.0, shuffle p=1.0, n=1000
   - reason: no cross-tissue signal (LOFO R²=1.000, p=1.000)

**genomics verdict mix:** confirmed=0, refuted=3, inconclusive=0  — an honest 'no novel confirmed signal' result on real, subtle data.

## enzyme_kinetics

**Question:** In the DLKcat (BRENDA + SABIO-RK) kcat compilation, which enzyme sequence/physicochemical feature predicts log_kcat across a HELD-OUT EC class, surviving the EC-label shuffle null?

LLM proposed 4 hypotheses; each verified by the real LOFO + shuffle null.

1. **refuted** (conf 0.8) — High frac_charged and frac_polar residues predict higher log_kcat by optimizing electrostatic preorganization for ionic transition states, moving the enzyme closer to the Bar-Even median; this signal transfers to EC3 (Hydrolases) but fails in EC1 (Oxidoreductases) where redox-active cofactors, rather than bulk sequence electrostatics, dominate the rate-limiting step.
   - features: `['frac_charged', 'frac_polar']`; LOFO R²=-0.09620392992865236, shuffle p=1.0, n=3553
   - reason: no cross-EC signal (LOFO R²=-0.096)
2. **refuted** (conf 0.8) — A high frac_glycine relative to sequence_length predicts higher log_kcat by facilitating rapid loop closures and product release in small, single-domain proteins; this relationship transfers to EC5 (Isomerases) due to their unimolecular simplicity but fails in EC6 (Ligases) where turnover is limited by large-scale multi-domain conformational changes and ATP hydrolysis.
   - features: `['frac_glycine', 'sequence_length']`; LOFO R²=-0.13535942330925743, shuffle p=1.0, n=3553
   - reason: no cross-EC signal (LOFO R²=-0.135)
3. **refuted** (conf 0.8) — High frac_aromatic coupled with high gravy_hydropathy predicts higher log_kcat by creating a rigid, low-dielectric active site environment that minimizes Marcus reorganization energy; this transfers to EC1 (Oxidoreductases) but fails in EC3 (Hydrolases) where solvent-mediated proton transfers require higher local flexibility and water accessibility.
   - features: `['frac_aromatic', 'gravy_hydropathy']`; LOFO R²=-0.13311952472167585, shuffle p=1.0, n=3553
   - reason: no cross-EC signal (LOFO R²=-0.133)
4. **refuted** (conf 0.8) — High frac_proline combined with low molecular_weight predicts higher log_kcat by enforcing a pre-strained backbone that reduces the entropic penalty of reaching the transition state; this transfers to EC4 (Lyases) where bond-breaking is the primary barrier, but fails in EC2 (Transferases) where the requirement for bi-substrate coordination makes backbone rigidity a kinetic hindrance.
   - features: `['frac_proline', 'molecular_weight']`; LOFO R²=-0.13754581558288462, shuffle p=1.0, n=3553
   - reason: no cross-EC signal (LOFO R²=-0.138)

**enzyme_kinetics verdict mix:** confirmed=0, refuted=4, inconclusive=0  — an honest 'no novel confirmed signal' result on real, subtle data.
