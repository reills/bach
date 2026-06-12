# ACE-Step 1.5 Integration

ACE-Step 1.5 is a downstream audio module in this repo. The v5 symbolic model owns notes, counterpoint, constraints, MusicXML, MIDI, and tablature. ACE-Step is only for audio styling, LoRA rendering, variation auditioning, or future audio reranking.

## Download

Do not vendor ACE-Step into this repository. Clone the official repo as an ignored external dependency and pin a release tag:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python scripts/setup_ace_step_15.py \
  --repo-dir /home/stephen-reilly/dev/ACE \
  --branch main \
  --install
```

The setup script defaults to the sibling checkout `/home/stephen-reilly/dev/ACE`, uses a shallow filtered `git clone` from `https://github.com/ace-step/ACE-Step-1.5.git`, tracks `origin/main` so normal `git pull` works, and optionally runs `uv sync` inside the ACE repo. It records `v0.1.8` as the currently tested release tag in generated manifests. It does not download model weights until ACE-Step itself is launched.

## Generate Symbolic Output With ACE Handoff

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python scripts/generate_instrumental_v5.py \
  --checkpoint out/instrumental_v5_overfit/checkpoint_latest.pt \
  --data-dir data/instrumental_v5/keyboard_overture_cnorm_outer2_v5 \
  --out-dir out/instrumental_v5_overfit/generated_guitar \
  --form invention \
  --key "D minor" \
  --subject "D4 E4 F4 A4 G4 F4 E4 D4" \
  --instrument classical_guitar \
  --hybrid-conditioning \
  --ace-step-handoff \
  --device cuda
```

This writes normal v5 artifacts plus:

- `ace_step_handoff/*.caption.txt`
- `ace_step_handoff/*.lyrics.txt`
- `ace_step_handoff/*.json`
- `*.ace_step_request.json`
- `ace_step_manifest.json`

The sidecar names match the ACE-Step 1.5 LoRA scanner convention. The expected audio file is listed in each JSON file. Render the generated MusicXML/MIDI to WAV/MP3 with a controlled sound before ACE-Step LoRA preprocessing.

## API Server

After setup, launch ACE-Step separately:

```bash
cd /home/stephen-reilly/dev/ACE
uv run acestep-api
```

The generated `*.ace_step_request.json` files are text-to-music request payloads for quick auditioning. They are not canonical score data and should not be transcribed back into notation without the symbolic checker.
