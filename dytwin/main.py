"""DyTwin DP+DM simulation entry point.

This open-source package runs the paper's main DyTwin setting:
dynamic profiling plus dynamic memory (DP+DM). Ablation and comparison
experiments are intentionally excluded from this lightweight release.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .data_loader import load_user_data
from .settings import settings
from .simulator import DynamicUserSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DyTwin DP+DM repost simulation.")
    parser.add_argument(
        "--user",
        required=True,
        help="User id, comma-separated ids, txt list, or 'all'.",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing SocialTwin user CSV files.")
    parser.add_argument("--result-dir", type=Path, default=None,
                        help="Directory for simulation outputs.")
    parser.add_argument("--start", default=None,
                        help="Start time, e.g. 2025-01-01 00:00:00.")
    parser.add_argument("--end", default=None,
                        help="End time, e.g. 2025-06-30 23:59:59.")
    parser.add_argument("--max-posts", type=int, default=None,
                        help="Maximum repost records to process from the earliest record.")
    parser.add_argument("--topk", type=int, default=None,
                        help="Memory retrieval top-k.")
    parser.add_argument("--overwrite", choices=["true", "false"], default="false",
                        help="Whether to overwrite existing user outputs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed passed to the LLM API when supported.")
    parser.add_argument("--no-llm-metrics", action="store_true",
                        help="Disable LLM-based evaluation metrics.")
    parser.add_argument("--llm-model", default=None,
                        help="Chat-completion model name.")
    parser.add_argument("--llm-api-url", default=None,
                        help="Chat-completion API URL.")
    parser.add_argument("--llm-api-key", default=None,
                        help="LLM API key. Defaults to DYTWIN_API_KEY.")
    parser.add_argument("--llm-temperature", type=float, default=None,
                        help="LLM decoding temperature.")
    parser.add_argument("--llm-max-tokens", type=int, default=None,
                        help="Maximum generation tokens.")
    parser.add_argument("--embedding-model-dir", type=Path, default=None,
                        help="Local BAAI/bge-small-zh-v1.5 SentenceTransformer directory.")
    parser.add_argument("--memory-similarity-threshold", type=float, default=None,
                        help="Memory retrieval similarity threshold.")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    if args.data_dir is not None:
        settings.user_data_dir = args.data_dir
    if args.result_dir is not None:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        settings.output_dir = args.result_dir
    if args.seed is not None:
        settings.seed = args.seed
    if args.max_posts is not None:
        settings.max_posts = args.max_posts
    if args.topk is not None:
        settings.memory_top_k = args.topk
    if args.no_llm_metrics:
        settings.use_llm_metrics = False
    if args.llm_model is not None:
        settings.llm_model = args.llm_model
    if args.llm_api_url is not None:
        settings.llm_api_url = args.llm_api_url
    if args.llm_api_key is not None:
        settings.llm_api_key = args.llm_api_key
    if args.llm_temperature is not None:
        settings.llm_temperature = args.llm_temperature
    if args.llm_max_tokens is not None:
        settings.llm_max_tokens = args.llm_max_tokens
    if args.embedding_model_dir is not None:
        os.environ["DYTWIN_EMBEDDING_MODEL_DIR"] = str(args.embedding_model_dir)
        settings._embedding_model = None
    if args.memory_similarity_threshold is not None:
        settings.memory_similarity_threshold = args.memory_similarity_threshold


def parse_user_list(user_arg: str, user_data_dir: Path) -> List[str]:
    user_arg = user_arg.strip()
    if user_arg.lower() == "all":
        users = [
            path.stem
            for path in user_data_dir.glob("*.csv")
            if path.name != "user_information.csv"
        ]
        return sorted(users)

    if user_arg.endswith(".txt"):
        txt_path = Path(user_arg)
        if not txt_path.is_absolute() and not txt_path.exists():
            txt_path = user_data_dir / user_arg
        if not txt_path.exists():
            raise FileNotFoundError(f"User list file not found: {user_arg}")
        return [
            line.strip()
            for line in txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if "," in user_arg:
        return [item.strip() for item in user_arg.split(",") if item.strip()]

    return [user_arg]


def filter_records(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    if args.max_posts is not None:
        return df.head(args.max_posts).reset_index(drop=True)

    if args.start:
        df = df[df["datetime"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["datetime"] <= pd.to_datetime(args.end)]
    return df.reset_index(drop=True)


def output_exists(user_id: str) -> bool:
    return (settings.output_dir / user_id / "simulation.csv").exists()


def run_user(user_id: str, args: argparse.Namespace) -> bool:
    if args.overwrite == "false" and output_exists(user_id):
        print(f"[skip] {user_id}: existing simulation.csv found.")
        return True

    try:
        df = load_user_data(settings.user_data_dir, user_id)
        df = filter_records(df, args)
        if len(df) == 0:
            print(f"[skip] {user_id}: no records after filtering.")
            return False

        print(f"\n[user] {user_id}: {len(df)} records")
        simulator = DynamicUserSimulator(
            settings=settings,
            user_id=user_id,
            overwrite=args.overwrite == "true",
        )
        simulator.config.memory_top_k = settings.memory_top_k
        simulator.run(df)
        print(f"[done] {user_id}: {settings.output_dir / user_id}")
        return True
    except Exception as exc:
        print(f"[error] {user_id}: {exc}")
        import traceback
        traceback.print_exc()
        return False


def print_configuration(args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    print("DyTwin DP+DM configuration")
    print("=" * 72)
    print(f"LLM model: {settings.llm_model}")
    print(f"LLM API URL: {settings.llm_api_url}")
    print(f"LLM API key: {'configured' if settings.llm_api_key else 'missing'}")
    print(f"Temperature: {settings.llm_temperature}")
    print(f"Max tokens: {settings.llm_max_tokens}")
    print(f"Seed: {settings.seed}")
    print(f"User data dir: {settings.user_data_dir}")
    print(f"Output dir: {settings.output_dir}")
    print(f"Embedding model dir: {settings.embedding_model_dir}")
    print(f"Memory top-k: {settings.memory_top_k}")
    print(f"Max posts: {args.max_posts if args.max_posts is not None else 'all filtered records'}")
    print("=" * 72 + "\n")


def main() -> None:
    args = parse_args()
    apply_args(args)

    if not settings.llm_api_key:
        print("[error] Missing LLM API key. Set DYTWIN_API_KEY or pass --llm-api-key.")
        return

    settings.apply_llm_config()
    if not settings.verify_llm_config():
        print("[error] LLM configuration verification failed.")
        return

    print_configuration(args)

    try:
        users = parse_user_list(args.user, settings.user_data_dir)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return

    if not users:
        print("[error] No users found.")
        return

    success = 0
    failed: List[str] = []
    for index, user_id in enumerate(users, 1):
        print(f"\n[{index}/{len(users)}] Processing {user_id}")
        if run_user(user_id, args):
            success += 1
        else:
            failed.append(user_id)

    print("\n" + "=" * 72)
    print("DyTwin run complete")
    print(f"Total: {len(users)}")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed users: {', '.join(failed)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
