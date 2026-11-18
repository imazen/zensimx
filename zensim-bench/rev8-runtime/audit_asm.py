#!/usr/bin/env python3
"""Verify the native Rev8 dense rows and any outlined helpers contain no packed division."""
import re
import subprocess
import sys


def main(executable):
    symbols = subprocess.check_output(["nm", "-nCS", executable], text=True)
    dense = re.compile(r"__attune_(terms|color|horizontal|vertical)_row|RowMath")
    checked = 0
    native = set()
    for line in symbols.splitlines():
        if "zensim::rev8::" not in line or not dense.search(line):
            continue
        start, size = line.split()[:2]
        stop = int(start, 16) + int(size, 16)
        asm = subprocess.check_output(
            ["objdump", "-d", "-C", "--no-show-raw-insn",
             f"--start-address=0x{start}", f"--stop-address={stop}", executable],
            text=True,
        )
        if re.search(r"\bv?divp[sd]\b", asm):
            raise SystemExit(f"Unexpected packed division in {line}")
        if row := re.search(r"__attune_(terms|color|horizontal|vertical)_row_(v3|v4x|v4)\b", line):
            required = "%ymm" if row[2] == "v3" else "%zmm"
            if required not in asm:
                raise SystemExit(f"Expected vector instructions in {line}")
            native.add(row.groups())
            print(f"{line} YMM={asm.count('%ymm')} ZMM={asm.count('%zmm')} packed divides=0")
        checked += 1
    if len(native) != 12:
        raise SystemExit(f"Missing native row symbols: {native}")
    print(f"{checked} dense symbols checked, zero packed divisions")


if __name__ == "__main__":
    main(sys.argv[1])
