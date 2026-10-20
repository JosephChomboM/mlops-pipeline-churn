"""churn: command line entry point.

churn init                  frozen split, train v1, promote it as the first champion
churn run-month --date ...  one monthly production run
churn simulate              init + six simulated months
churn status                current champion and its decision policy
churn model-card            regenerate reports/model_card.md
"""

import argparse
import json

from churn import config
from churn.models.bundle import load_bundle
from churn.pipelines import orchestrator
from churn.reporting.model_card import write_model_card


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="churn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    month = sub.add_parser("run-month")
    month.add_argument("--date", required=True)
    month.add_argument("--shift", type=float, default=0.0)
    month.add_argument("--corrupt", action="store_true")
    sub.add_parser("simulate")
    sub.add_parser("status")
    sub.add_parser("model-card")
    args = parser.parse_args()

    if args.command == "init":
        orchestrator.reset()
        orchestrator.init()
    elif args.command == "run-month":
        orchestrator.run_month(args.date, args.shift, args.corrupt)
    elif args.command == "simulate":
        orchestrator.simulate()
    elif args.command == "model-card":
        write_model_card()
    else:
        bundle = load_bundle(config.CHAMPION_DIR)
        print(json.dumps({"champion": bundle["metadata"], "policy": bundle["policy"]}, indent=2))


if __name__ == "__main__":
    main()
