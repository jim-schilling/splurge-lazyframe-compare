import argparse
from splurge_lazyframe_compare.utils.logging_helpers import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slc",
        description="Splurge LazyFrame Compare CLI",
    )
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="command", required=True)

    # compare subcommand
    compare = sub.add_parser("compare", help="Compare two LazyFrames")
    compare.add_argument("--dry-run", action="store_true", help="Validate inputs and exit")

    # report subcommand
    report = sub.add_parser("report", help="Generate report from comparison results")
    report.add_argument("--dry-run", action="store_true", help="Validate inputs and exit")

    # export subcommand
    export = sub.add_parser("export", help="Export comparison results to files")
    export.add_argument("--dry-run", action="store_true", help="Validate inputs and exit")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Configure logging for CLI runs
    configure_logging(level=args.log_level)

    if args.command == "compare":
        if args.dry_run:
            print("Dry run: compare would execute with provided parameters")
            return 0
        print("Compare command requires input parameters in future iteration")
        return 2
    if args.command == "report":
        if args.dry_run:
            print("Dry run: report would generate summary/detailed report")
            return 0
        print("Report command requires input parameters in future iteration")
        return 2
    if args.command == "export":
        if args.dry_run:
            print("Dry run: export would write results to output directory")
            return 0
        print("Export command requires input parameters in future iteration")
        return 2

    return 0


