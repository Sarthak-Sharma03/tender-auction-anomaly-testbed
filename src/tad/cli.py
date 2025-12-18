from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .quickstart import run_quickstart
from .run_config import run_from_yaml


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tad",
        description=(
            "Tender/Auction Anomaly Testbed "
            "(decision-centric pipeline + reference implementation)."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    qs = sub.add_parser(
        "quickstart",
        help=(
            "Run the end-to-end pipeline and write a self-contained report folder."
        ),
    )
    qs.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the run (e.g., reports/quickstart).",
    )
    qs.add_argument(
        "--input",
        default=None,
        help=(
            "Optional input file path (xlsx/csv). "
            "If omitted, uses the repo sample file."
        ),
    )
    qs.add_argument(
        "--budgets",
        default="0.005,0.01,0.05",
        help=(
            "Comma-separated review budgets (fractions), "
            "e.g. 0.005,0.01,0.05"
        ),
    )
    qs.add_argument(
        "--time-split",
        type=float,
        default=0.7,
        help="Train/test split fraction by time (0..1). Default: 0.7",
    )
    qs.add_argument(
        "--random-state",
        type=int,
        default=13,
        help="Random seed. Default: 13",
    )
    qs.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in synthetic generator instead of a file.",
    )
    qs.add_argument(
        "--ensemble",
        default="fisher",
        choices=["fisher", "rank"],
        help=(
            "Primary ensemble method for reports/top sessions "
            "(fisher = calibrated p-value aggregation)."
        ),
    )

    rc = sub.add_parser("run-config", help="Run a YAML experiment config.")
    rc.add_argument(
        "config",
        help="Path to a YAML file under experiments/configs/ or similar.",
    )

    api = sub.add_parser(
        "serve-api",
        help="Serve a reference scoring API (FastAPI) from a model bundle.",
    )
    api.add_argument("--bundle", required=True, help="Path to model_bundle/ directory.")
    api.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind. Default: 127.0.0.1",
    )
    api.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind. Default: 8000",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "quickstart":
        budgets = tuple(
            float(x.strip()) for x in str(args.budgets).split(",") if x.strip()
        )
        data_path = str(args.input) if args.input else "data/Login_Data.xlsx"
        run_quickstart(
            data_path=data_path,
            out_dir=Path(args.out_dir),
            budgets=budgets,
            time_split=float(args.time_split),
            random_state=int(args.random_state),
            synthetic=bool(args.synthetic),
            ensemble_method=str(args.ensemble),
        )
        return 0

    if args.cmd == "run-config":
        run_from_yaml(Path(args.config))
        return 0

    if args.cmd == "serve-api":
        try:
            import uvicorn
        except Exception as e:
            print(f"uvicorn is required to serve the API: {e}", file=sys.stderr)
            return 2

        from .api.app import create_app

        app = create_app(Path(args.bundle))
        print(f"[tad] serving: bundle={args.bundle} http://{args.host}:{args.port}")
        uvicorn.run(app, host=str(args.host), port=int(args.port))
        return 0

    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
