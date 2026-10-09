"""Command-line interface.

Examples::

    oulad-analytics train --data-dir path/to/OULAD --cutoff 70 --out-dir results
    oulad-analytics features --data-dir path/to/OULAD --out features.csv
    oulad-analytics demo --out-dir results-demo     # synthetic data, no download
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import __version__
from .features import build_feature_table
from .loader import SchemaError, load_oulad
from .model import feature_importance, train_and_evaluate
from .synthetic import write_synthetic
from .viz import plot_feature_importance, plot_metrics, plot_weekly_clicks


def _markdown_table(df) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _run_training(
    data_dir: Path, cutoff: int, out_dir: Path, seed: int, label: str | None = None
) -> Path:
    data = load_oulad(data_dir)
    table = build_feature_table(data, cutoff_day=cutoff)
    result, split = train_and_evaluate(table, seed=seed)
    importance = feature_importance(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.round(4).to_csv(out_dir / "metrics.csv", index=False)
    importance.round(4).to_csv(out_dir / "feature_importance.csv", header=["importance"])
    plot_weekly_clicks(table, out_dir / "weekly_clicks.png")
    plot_feature_importance(importance, out_dir / "feature_importance.png")
    plot_metrics(result.metrics, out_dir / "model_comparison.png")

    test = result.metrics[result.metrics["split"] == "test"].drop(columns="split")
    sizes = split.sizes()
    summary = [
        "# Baseline results",
        "",
        f"- data: {label or f'`{data_dir}`'}",
        f"- cut-off day: {cutoff}",
        f"- enrolments still active at cut-off: {len(table)}",
        f"- at-risk share: {table['at_risk'].mean():.1%}",
        f"- split (grouped by student): train {sizes['train']}, "
        f"val {sizes['val']}, test {sizes['test']}",
        "",
        "## Test set",
        "",
        _markdown_table(test.round(3)),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oulad-analytics", description="Early prediction of at-risk students (OULAD)."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_feat = sub.add_parser("features", help="build the feature table and save it as CSV")
    p_feat.add_argument("--data-dir", required=True, type=Path)
    p_feat.add_argument("--cutoff", type=int, default=70, help="cut-off day (default 70)")
    p_feat.add_argument("--out", type=Path, default=Path("features.csv"))

    p_train = sub.add_parser("train", help="train and evaluate the baseline models")
    p_train.add_argument("--data-dir", required=True, type=Path)
    p_train.add_argument("--cutoff", type=int, default=70)
    p_train.add_argument("--out-dir", type=Path, default=Path("results"))
    p_train.add_argument("--seed", type=int, default=42)

    p_syn = sub.add_parser("synthetic", help="write a synthetic OULAD-format dataset")
    p_syn.add_argument("--out", type=Path, required=True)
    p_syn.add_argument("--n-students", type=int, default=400)
    p_syn.add_argument("--seed", type=int, default=0)

    p_demo = sub.add_parser("demo", help="run the full pipeline on synthetic data")
    p_demo.add_argument("--out-dir", type=Path, default=Path("results-demo"))
    p_demo.add_argument("--n-students", type=int, default=600)
    p_demo.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "features":
            table = build_feature_table(load_oulad(args.data_dir), cutoff_day=args.cutoff)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(f"Saved {len(table)} rows x {table.shape[1]} columns to {args.out}")
        elif args.command == "train":
            _run_training(args.data_dir, args.cutoff, args.out_dir, args.seed)
        elif args.command == "synthetic":
            write_synthetic(args.out, args.n_students, args.seed)
            print(f"Synthetic OULAD-format data written to {args.out}")
        elif args.command == "demo":
            with tempfile.TemporaryDirectory() as tmp:
                write_synthetic(tmp, args.n_students, seed=0)
                _run_training(Path(tmp), 70, args.out_dir, args.seed, label="synthetic demo data")
    except SchemaError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
