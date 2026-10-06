"""Isolated experiment process used by the bounded run scheduler."""

import sys
from pathlib import Path

from speculators.generator.engine import worker

if __name__ == "__main__":
    worker(Path(sys.argv[1]))
