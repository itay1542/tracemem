"""CLI entry point for the TraceMem installer."""

import argparse
import sys

from tracemem_installer.install import run_init
from tracemem_installer.uninstall import run_uninstall


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracemem-claude",
        description="TraceMem — memory hooks installer for Claude Code",
    )
    sub = parser.add_subparsers(dest="command")

    init_parser = sub.add_parser("init", help="Install TraceMem hooks")
    init_parser.add_argument(
        "--global",
        dest="scope",
        action="store_const",
        const="global",
        default="local",
        help="Install to ~/.claude/ (default: ./.claude/)",
    )
    init_parser.add_argument(
        "--local",
        dest="scope",
        action="store_const",
        const="local",
        help="Install to ./.claude/ (default)",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing installation",
    )

    uninstall_parser = sub.add_parser("uninstall", help="Remove TraceMem hooks")
    uninstall_parser.add_argument(
        "--global",
        dest="scope",
        action="store_const",
        const="global",
        default="local",
        help="Uninstall from ~/.claude/ (default: ./.claude/)",
    )
    uninstall_parser.add_argument(
        "--local",
        dest="scope",
        action="store_const",
        const="local",
        help="Uninstall from ./.claude/ (default)",
    )

    args = parser.parse_args()

    if args.command == "init":
        run_init(args.scope, force=args.force)
    elif args.command == "uninstall":
        run_uninstall(args.scope)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
