# Reproduction Guide

This guide covers the lightweight DyTwin DP+DM simulation pipeline.

## 1. Prepare Data

Place the full SocialTwin CSV files under:

```text
data/SocialTwin/
```

The command `--user all` scans this directory and treats every CSV filename stem as a user id.

## 2. Configure API Access

Set the LLM API key:

```bash
set DYTWIN_API_KEY=your_api_key_here
```

The default chat-completion endpoint and model can be overridden:

```bash
python -m dytwin.main --llm-model MODEL --llm-api-url URL --llm-api-key KEY ...
```

## 3. Prepare the Local Embedding Model

DyTwin follows the paper configuration and uses local
`BAAI/bge-small-zh-v1.5` embeddings. Place the SentenceTransformer
checkpoint under:

```text
models/BAAI/bge-small-zh-v1.5/BAAI/bge-small-zh-v1___5/
```

If the checkpoint is stored elsewhere, set:

```bash
set DYTWIN_EMBEDDING_MODEL_DIR=path\to\bge-small-zh-v1___5
```

## 4. Run DP+DM

```bash
python -m dytwin.main --user all --data-dir data/SocialTwin --result-dir outputs --max-posts 100 --overwrite false
```

This creates one folder per user:

```text
outputs/<user_id>/simulation.csv
outputs/<user_id>/memory/
outputs/<user_id>/visualization/
```

## 5. Smoke Test

```bash
python -m dytwin.main --user all --data-dir examples/sample_data --result-dir outputs/sample_run --max-posts 20 --overwrite true
```

Use the smoke test to check installation, paths, API access, local embedding loading, and output generation. It is not intended to reproduce paper-scale aggregate statistics.
