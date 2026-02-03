#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tracemem-core>=0.1.0", "pyyaml>=6.0"]
# ///
"""TraceMem Claude Code hook entry point.

UV resolves tracemem-core + transitive deps from PyPI into a cached venv.
Subsequent runs reuse the cache — no pip install into the project needed.
"""

import sys
from pathlib import Path

# Add the skill directory to sys.path so tracemem_claude can be imported
sys.path.insert(0, str(Path(__file__).parent))

from tracemem_claude.cli import main

main()
