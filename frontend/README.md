# Bach Gen Frontend

React + Vite + AlphaTab UI for composing and inpainting guitar scores.

## Run locally

```bash
npm install
npm run dev
```

Optional API base override:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Local test-data mode (no backend required)

Place your MusicXML files under `frontend/public/test-data/` and add a
`manifest.json`:

```json
{
  "baseScore": "base.musicxml",
  "snippets": [
    "measure-001.xml",
    "measure-002.xml"
  ]
}
```

- `baseScore` should be a full MusicXML score.
- `snippets` can be full MusicXML files or raw `<measure>` fragments.
- Use the UI data source selector to switch to `Local test-data`, then click
  `Load test-data`.

To default into local mode:

```bash
VITE_USE_LOCAL_DATA=true npm run dev
```

## Notes

- AlphaTab renders score + tab from the MusicXML you load.
- In local mode, inpainting swaps the selected measure with a random snippet.
