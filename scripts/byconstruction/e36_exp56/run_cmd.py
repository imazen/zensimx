#!/usr/bin/env python3
"""Run a continuation command recorded as JSON {argv, env}: the recipe's argv with its env and paths substituted."""
import json, os, subprocess, sys
c = json.load(open(sys.argv[1]))
env = dict(os.environ); env.update(c['env'])
sys.exit(subprocess.run(c['argv'], env=env).returncode)
