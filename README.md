# DyTwin

DyTwin is the open-source implementation of the main social digital twin simulation pipeline described in our manuscript (currently under review). This lightweight release focuses on the core DP+DM setting: dynamic profiling plus dynamic memory.

It does not include paper drafting files, rebuttal materials, human/Turing-test scripts, ablation experiment code, full experiment outputs, local model checkpoints, or the full SocialTwin dataset.

## Repository Layout

```text
DyTwin/
|-- dytwin/                  # Core simulation and evaluation package
|-- configs/                 # Configuration template
|-- docs/                    # Dataset and reproduction notes
|-- examples/sample_data/    # Small sample CSVs for smoke tests
|-- data/                    # Dataset placement instructions only
|-- outputs/                 # Generated results, ignored by git
`-- requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set your LLM API key before running simulations:

```bash
set DYTWIN_API_KEY=your_api_key_here
```

On Linux/macOS:

```bash
export DYTWIN_API_KEY=your_api_key_here
```

## Local Embedding Model

DyTwin uses the same local embedding model as the paper experiments: `BAAI/bge-small-zh-v1.5`.

Place the SentenceTransformer model directory at:

```text
models/BAAI/bge-small-zh-v1.5/BAAI/bge-small-zh-v1___5/
```

If the model is stored elsewhere, set:

```bash
set DYTWIN_EMBEDDING_MODEL_DIR=path\to\bge-small-zh-v1___5
```

On Linux/macOS:

```bash
export DYTWIN_EMBEDDING_MODEL_DIR=/path/to/bge-small-zh-v1___5
```

## Data

The full `SocialTwin` dataset (about 616 MB) is not committed to GitHub. Download it from Zenodo (DOI: 10.5281/zenodo.22218365) and place user CSV files under:

```text
data/SocialTwin/
```

For quick smoke tests, two small CSV files are included under `examples/sample_data/`.

## Run DyTwin

Run DP+DM on the full dataset:

```bash
python -m dytwin.main --user all --data-dir data/SocialTwin --result-dir outputs --max-posts 100 --overwrite false
```

Run a smoke test on the sample users:

```bash
python -m dytwin.main --user all --data-dir examples/sample_data --result-dir outputs/sample_run --max-posts 20 --overwrite true
```

Each user output is written to:

```text
outputs/<user_id>/simulation.csv
outputs/<user_id>/memory/
outputs/<user_id>/visualization/
```

The generated CSV contains predictions, retrieved memories, profile updates, ROUGE-L, BERTScore, embedding similarity, and LLM-based evaluation scores.

## Notes

- Generated outputs, vector indexes, logs, and model checkpoints are ignored by git.
- DyTwin uses local `BAAI/bge-small-zh-v1.5` embeddings by default to match the paper configuration.
- BERTScore uses the same local `BAAI/bge-small-zh-v1.5` checkpoint with four layers, matching the paper configuration.
- Do not commit API keys or private experiment outputs.
