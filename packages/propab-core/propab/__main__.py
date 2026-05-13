"""Allow ``python -m propab health`` without a console_scripts install."""

from propab.cli import main

if __name__ == "__main__":
    main()
