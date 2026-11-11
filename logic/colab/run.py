#!/usr/bin/env python3
"""The research moved to research_m5/. This file only passes a run started from the old place on to the new one."""
import os
import runpy
import sys

new = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "research_m5", "colab", "run.py")
sys.argv[0] = os.path.abspath(new)
runpy.run_path(sys.argv[0], run_name="__main__")
