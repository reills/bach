# MusicXML Cleaning

The raw corpus under `data/tobis_xml` is immutable. The cleaner creates a separate
MusicXML mirror with source hashes, repair records, duplicate reports, and a review
queue.

Automatic repair is deliberately conservative:

- It compares declared meters with actual bar durations across all parsed staves.
- It repairs short, isolated meter runs only when the surrounding movement has a
  strong dominant meter and the note content agrees with that meter.
- It does not guess between meters with the same duration, such as `3/4` and `6/8`.
- It quarantines isolated equal-duration changes, such as `4/4` inside an `8/8`
  run, until an authoritative score resolves the intended beat grouping.
- A reviewed meter override still fails approval when internal bar durations
  conflict with the corrected meter; meter edits cannot hide corrupt note data.
- It preserves legitimate short measures, pickups, and meter changes when their
  content agrees with the declared signature.

Build the default counterpoint-oriented mirror:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u \
  scripts/clean_musicxml_corpus.py \
  --output-dir data/musicxml_cleaned/bach_counterpoint_v1
```

The default profile covers keyboard works, Art of Fugue, Canons, Musical Offering,
and selected organ trio/fugue collections. Use repeated `--source` arguments to
build a narrower mirror.

Outputs:

- `files/`: copied or repaired MusicXML files, retaining source-relative paths.
- `manifest.jsonl`: SHA-256 provenance, meters, changes, and issues per file.
- `approved_files.txt`: files that passed the meter-integrity gate.
- `review_queue.json`: ambiguous files excluded from automatic approval.
- `duplicates.json`: exact source-byte duplicates.
- `meter_overrides.snapshot.json`: the reviewed overrides used for the run.

## Reviewed Overrides

Whole-movement meter identity sometimes requires an authoritative score. Add a
reviewed entry to `configs/musicxml_meter_overrides.json`:

```json
{
  "path": "instrumental-works/keyboard-works/example/BWV_example.xml",
  "movement_index": 0,
  "time_signature": "4/4",
  "start_measure": 0,
  "end_measure": null,
  "source": "edition/catalog citation",
  "note": "The encoded 3/4 change is not present in the reviewed score."
}
```

`movement_index` is zero-based. `start_measure` and `end_measure` are relative to
that movement; a null end applies through the movement. Evidence is stored in the
snapshot for reproducibility.

Meter approval is not the final corpus gate. Voice count, monophonic lane quality,
crossings, duplicate works, and source family still need the v6 dataset audit before
training.
