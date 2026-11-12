"""Hold the study-scoped idle/sleep inhibitor until verified closure."""
import argparse
import time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--marker",type=Path,required=True)
args=p.parse_args()
while not args.marker.is_file():
    time.sleep(60)
