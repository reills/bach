# Time Signature Audit (Tobis vs music21 corpus)

Script:

- `data-testing/audit_tobis_time_sigs_with_music21.py`

What it does:

- Scans Tobis `.xml` files.
- Flags likely noisy meter metadata:
  - equivalent meter flips (example: `4/4 -> 8/8`)
  - isolated one-measure equivalent blips (`A -> B -> A`)
- Optionally compares each BWV file to `music21.corpus` Bach entries (default: only chorales).
- Extracts XML-level time signature usage across all parts.
- Computes tuplet stats (3:2 triplets vs total notes) and tags a derived `compound_feel` flag.

## Run

From project root:

```bash
python data-testing/audit_tobis_time_sigs_with_music21.py
```

Default scan is full Tobis corpus under `data/tobis_xml`.

Quick smoke test:

```bash
python data-testing/audit_tobis_time_sigs_with_music21.py --limit 20 --progress-every 10
```

Disable corpus comparison:

```bash
python data-testing/audit_tobis_time_sigs_with_music21.py --trusted-scope none
```

Adjust the compound-feel threshold (triplet 3:2 ratio over total notes):

```bash
python data-testing/audit_tobis_time_sigs_with_music21.py \
  --compound-triplet-threshold 0.4
```

Custom directories (your two folders):

```bash
python data-testing/audit_tobis_time_sigs_with_music21.py \
  --dirs "data/tobis_xml/vocal-works/Cantatas" "data/tobis_xml/vocal-works/chorales"
```

## Output files

Default output prefix is `data-testing/time_sig_audit`, producing:

- `data-testing/time_sig_audit_files.csv` (one row per file)
- `data-testing/time_sig_audit_summary.json` (global counts)

Key CSV columns:

- `flag_suspicious`: true if equivalent flip and/or isolated equivalent blip found
- `equivalent_event_changes`
- `isolated_equivalent_blips`
- `trusted_*` columns for music21 corpus comparison
- `xml_unique_sigs`: unique time signatures found directly in XML (all parts)
- `xml_total_notes`, `xml_time_mod_notes`, `xml_triplet_3_2_notes`
- `xml_triplet_3_2_ratio`: `triplet_3_2_notes / total_notes`
- `compound_feel`: `xml_triplet_3_2_ratio >= compound_triplet_threshold`

## Dependency

```bash
pip install music21
```
