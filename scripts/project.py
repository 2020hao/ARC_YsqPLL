"""Source-checkout entry point for the current V2 project CLI."""

import _bootstrap  # noqa: F401
from arc_model.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
