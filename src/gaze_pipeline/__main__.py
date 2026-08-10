"""Allow ``python -m gaze_pipeline`` to invoke the public CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
