from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginshield.tournament import TournamentConfig, load_dataset, tournament, train_live_model, validate_policy_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MarginShield's coordinated refund-abuse ring scorer")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-precision", type=float, default=0.85,
        help="Reported comparison baseline only. Not a constraint: both tiers are selected by net value.",
    )
    parser.add_argument(
        "--minimum-validation-flags", type=int, default=30,
        help="Flag count used with --minimum-precision for the reported comparison baseline.",
    )
    parser.add_argument("--maximum-manual-reviews", type=int, default=100)
    parser.add_argument("--minimum-manual-reviews", type=int, default=30)
    parser.add_argument("--maximum-verifications", type=int, default=100)
    parser.add_argument("--minimum-verifications", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--tournament-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.minimum_precision <= 1:
        parser.error("--minimum-precision must be in (0, 1]")
    config = TournamentConfig(
        minimum_precision=args.minimum_precision,
        minimum_validation_flags=args.minimum_validation_flags,
        maximum_manual_reviews=args.maximum_manual_reviews,
        minimum_manual_reviews=args.minimum_manual_reviews,
        maximum_verifications=args.maximum_verifications,
        minimum_verifications=args.minimum_verifications,
        bootstrap_samples=args.bootstrap_samples,
    )
    if args.tournament_only:
        winner, report = tournament(load_dataset(args.dataset), config)
        print(json.dumps(report, indent=2))
        return
    if args.validation_only:
        report = validate_policy_only(load_dataset(args.dataset), config)
        print(json.dumps(report, indent=2))
        return
    report = train_live_model(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        config=config,
    )
    print(json.dumps({
        "winner": report["winner"],
        "policy_validation": report["policy_validation"],
        "test": report["test"],
        "test_temporal_bootstrap_95pct_ci": report["test_temporal_bootstrap_95pct_ci"],
    }, indent=2))


if __name__ == "__main__":
    main()
