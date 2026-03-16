# STAMP + MoPaDi GUI

Interactive web interface for exploring STAMP MIL classifier predictions and MoPaDi counterfactual explanations on whole-slide images.

## Workflow

1. **Tiling & feature extraction** — enter a WSI path, select a feature extractor, run STAMP preprocessing
2. **Classifier + heatmap** — run a trained STAMP classifier, view the attention heatmap and top predictive tiles
3. **Counterfactuals** — click tiles to select them, choose an amplitude, generate MoPaDi counterfactuals

## Setup


Install additionaldependencies into the mopadi venv:

```bash
source mopadi/.venv/bin/activate
pip install fastapi uvicorn jinja2 python-multipart
```

## Running

```bash
source mopadi/.venv/bin/activate
python gui/app.py
```

Then open `http://localhost:3010` in your browser. If running on a remote server, forward the port:

```bash
ssh -L 3010:localhost:3010 user@server
```

## Configuration

All server-specific paths live in `.env`. Required variables:

| Variable | Description |
|---|---|
| `AUTOENC_CKPT` | Path to the MoPaDi autoencoder checkpoint |
| `MOPADI_CONFIG_PATH` | Path to the MoPaDi YAML config |
| `CLASSIFIER_BRAF_CKPT` | Path to STAMP BRAF classifier checkpoint |
| `CLASSIFIER_MSI_CKPT` | Path to STAMP MSI classifier checkpoint |
| `EXAMPLE_WSI_PATH` | Optional: pre-filled example WSI path in the UI |

## Files

```
gui/
├── app.py               # FastAPI app — routes and event wiring
├── stamp_runner.py      # STAMP tiling, feature extraction, heatmap wrappers
├── mopadi_runner.py     # MoPaDi ImageManipulatorSTAMP wrapper + model pool
├── utils.py             # Thumbnail, tile overlay, base64, comparison slider HTML
├── templates/
│   └── index.html       # Single-page UI
├── .env                 # Server-specific paths
├── .env.example         # Template for .env
└── .gitignore
```
