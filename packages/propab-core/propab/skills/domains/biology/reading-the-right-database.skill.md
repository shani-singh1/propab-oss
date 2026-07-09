---
name: reading-the-right-database
description: Query the authoritative primary database for the entity or claim at hand — UniProt for proteins, Ensembl/NCBI for genes and variants, PDB/AlphaFold for structures, KEGG/Reactome for pathways, GEO/GTEx for expression — and carry accession, genome build, and version as provenance
phase: hypothesis
scope: biology
priority: 15
---
A biological fact is only as trustworthy as the source you pulled it from and the version you pulled.
Using the wrong database, an aggregator's stale mirror, or the wrong genome build silently corrupts
everything downstream. Know which resource is authoritative for each kind of claim, and record enough
provenance that the lookup is reproducible.

1. **Match the entity to its authoritative source.**
   - Proteins (sequence, function, domains, PTMs, isoforms) → UniProt (note Swiss-Prot *reviewed* vs
     TrEMBL *unreviewed*).
   - Genes / transcripts / reference genome → Ensembl or NCBI Gene / RefSeq. Variant frequency →
     gnomAD / dbSNP; clinical significance → ClinVar.
   - Structures → PDB for EXPERIMENTAL structures; AlphaFold DB for PREDICTED ones (report pLDDT and
     label it predicted).
   - Pathways / reactions → KEGG, Reactome. Protein interactions → STRING (carry the confidence
     score; STRING mixes predicted and measured edges).
   - Expression → GEO / ArrayExpress (raw studies), GTEx (normal tissue), TCGA (cancer). Ontology /
     gene sets → GO, MSigDB.

2. **Record provenance and VERSION.** Store the accession/ID, the database release (e.g. Ensembl
   110), the genome build (GRCh38 vs GRCh37/hg19 — a mismatch shifts every coordinate and is a common
   silent catastrophe), and the access date. A claim without its accession and build is not
   reproducible and cannot be re-checked.

3. **Map identifiers through stable accessions, not symbols.** Gene symbols are ambiguous, aliased,
   and deprecated over time; map via stable IDs (Ensembl / UniProt accessions). Beware spreadsheet
   auto-conversion mangling symbols (SEPT2 → a date), which silently deletes genes from an analysis.

4. **Distinguish measured from inferred.** A predicted structure, a STRING functional-association
   edge, or a GO term tagged IEA (inferred from electronic annotation) is weaker than an
   experimentally determined one. Do not cite an inference as a measurement.

5. **Absence is not evidence of absence.** No record in a database means "not annotated here", not
   "does not exist". Report the gap as a gap rather than concluding the negative.

The dishonest move is presenting a database value without its provenance, citing a predicted or
electronically-inferred annotation as an established fact, or letting a genome-build or symbol-mapping
error pass unremarked. The bar: every retrieved fact carries its authoritative source, accession,
version/build, and access date, and is labeled measured or inferred — so a third party can pull the
identical record and be forced to the same claim.
