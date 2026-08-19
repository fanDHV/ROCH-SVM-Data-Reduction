"""Command-line entry point for a reproducible ROCH-SVM experiment."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare standard SVM, ROCH-SVM, and CCH-SVM on a CSV dataset."
    )
    parser.add_argument("dataset", help="Dataset stem under data/ (for example: haberman)")
    parser.add_argument("--runs", type=int, default=1, help="Number of stratified train/test runs")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("svm", "roch", "cch"),
        default=("svm", "roch", "cch"),
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument("--c", type=float, default=5.0, help="SVM C used by all selected methods")
    parser.add_argument("--gamma", type=float, default=2.0, help="RBF gamma used by all methods")
    parser.add_argument("--ha", type=int, default=4, help="High-dimensional partition depth")
    parser.add_argument("--hb", type=int, default=10, help="Boundary-extraction depth")
    parser.add_argument("--hg", type=int, default=12, help="Grid depth")
    parser.add_argument("--k", type=int, nargs="+", default=[9], help="CCH neighborhood sizes")
    parser.add_argument("--l-factor", type=float, default=0.05)
    parser.add_argument("--min-points", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Import after parsing so `--help` remains available before dependencies
    # are installed and the CLI module stays lightweight.
    from Train_SVM_v3_flag import run

    methods = set(args.methods)
    run(
        args.dataset,
        run_original="svm" in methods,
        run_roch="roch" in methods,
        run_cch="cch" in methods,
        C_org=args.c,
        gamma_org=args.gamma,
        C_roch=args.c,
        gamma_roch=args.gamma,
        C_cch=args.c,
        gamma_cch=args.gamma,
        ha_roch=args.ha,
        hb_roch=args.hb,
        hg_roch=args.hg,
        ha_cch=args.ha,
        hb_cch=args.hb,
        hg_cch=args.hg,
        K_values=args.k,
        L_factor=args.l_factor,
        min_points=args.min_points,
        n_runs=args.runs,
    )


if __name__ == "__main__":
    main()
