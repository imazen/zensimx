"""Selected native uncertainty readers for a future registered confidence trial.

Conversions follow zenpapers release_spreads.py at 4e81c03a46fe2a9d86a73c3da75edaaec7fd9b9c.
Selection and source pins must come from a separately frozen fitting manifest.
These readers do not admit a population, construct weights, or change targets.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
from subjective_training_leg import safe


def digest(path):
    with safe(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def authorize(ledger, source, pin, wanted, columns, protocol):
    """Opaque integrity reads are recorded separately from selected values."""
    source, ledger = safe(source), safe(ledger)
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("nonempty unique frozen selection required")
    d = json.loads(ledger.read_text())
    if d.get("role") != "TRAIN" or not d.get("fitting_manifest_sha256"):
        raise ValueError("independent fitting population authority required")
    d["reads"].append(dict(kind="opaque-source-integrity",path=str(source),sha256=pin))
    ledger.write_text(json.dumps(d,indent=1)+"\n")
    if digest(source) != pin:
        raise ValueError("native uncertainty source changed")
    d["reads"].append(dict(kind="selected-native-uncertainty",path=str(source),sha256=pin,
        selected_items=sorted(wanted),columns=columns,protocol=protocol,state="authorized-before-read"))
    ledger.write_text(json.dumps(d,indent=1)+"\n")
    return source


def complete(ledger, wanted, output):
    if set(output) != set(wanted):
        raise ValueError("selected uncertainty population incomplete")
    d = json.loads(ledger.read_text())
    d["reads"][-1].update(state="verified",rows=len(output))
    ledger.write_text(json.dumps(d,indent=1)+"\n")
    return output


def nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("finite nonnegative native uncertainty required")
    return result


def kadid(path, pin, wanted, ledger):
    """Release IDs first; parse only selected variance, never the DMOS value."""
    path = authorize(ledger,path,pin,wanted,["dist_img","var"],"KADID variance / 30 DCRs")
    result = {}
    with path.open(newline="") as stream:
        header = next(csv.reader([next(stream)]))
        image, variance = header.index("dist_img"),header.index("var")
        if image != 0:
            raise ValueError("native KADID identifier must be the first column")
        for line in stream:
            # Native IDs have no commas. Do not tokenize an excluded response record.
            item = next(csv.reader([line.partition(",")[0]]))[0].lower()
            if item not in wanted:
                continue
            if item in result:
                raise ValueError("duplicate selected native item")
            row = next(csv.reader([line]))
            if len(row) != len(header):
                raise ValueError("native KADID row shape differs")
            var = nonnegative(row[variance])
            result[item] = dict(se_native=math.sqrt(var/30),n=30,source="released-observer-variance")
    return complete(ledger,wanted,result)


def tid(names, names_pin, spreads, spreads_pin, wanted, ledger):
    """Published uncertainty of MOS; no second division by observer count."""
    names = authorize(ledger,names,names_pin,wanted,["image-name"],"TID native row identity")
    selected, count = {}, 0
    with names.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError("TID name row shape differs")
            item = fields[1].lower()
            if item in wanted:
                if item in selected.values():
                    raise ValueError("duplicate selected native item")
                selected[count] = item
            count += 1
    if set(selected.values()) != set(wanted):
        raise ValueError("selected TID name population incomplete")
    d = json.loads(ledger.read_text())
    d["reads"][-1].update(state="verified",rows=len(selected),source_rows=count)
    ledger.write_text(json.dumps(d,indent=1)+"\n")
    spreads = authorize(ledger,spreads,spreads_pin,wanted,["MOS-standard-error"],"TID released MOS uncertainty")
    result, total = {}, 0
    with spreads.open() as stream:
        for line in stream:
            for token in line.split():
                if total in selected:
                    result[selected[total]] = dict(se_native=nonnegative(token),n=None,source="released-MOS-standard-error")
                total += 1
    if total != count:
        raise ValueError("TID uncertainty/name row count differs")
    return complete(ledger,wanted,result)


def published_spread(path, pin, wanted_rows, ledger):
    """Normalized released item SD/count. No implicit observer-model fallback.

    wanted_rows maps frozen native row indices to exact relative item paths.
    Caller has already bound the normalized table and its importer provenance.
    """
    path = authorize(ledger,path,pin,list(wanted_rows.values()),["item","std","n_ratings"],
        "released-observer-spread / sqrt(native count)")
    if any(type(i) is not int or i < 0 for i in wanted_rows):
        raise ValueError("frozen native row indices required")
    result = {}
    with path.open() as stream:
        header = next(csv.reader([next(stream)],delimiter="\t"))
        positions = [header.index(c) for c in ("item","std","n_ratings")]
        for i,line in enumerate(stream):
            if i not in wanted_rows:
                continue
            fields = next(csv.reader([line],delimiter="\t"))
            if len(fields) != len(header):
                raise ValueError("normalized uncertainty row shape differs")
            item,sd,n = (fields[j] for j in positions)
            if item != wanted_rows[i] or item in result:
                raise ValueError("native uncertainty item binding differs")
            sd,n = nonnegative(sd),nonnegative(n)
            if n < 2 or n != int(n):
                raise ValueError("native observer count must be integer and at least two")
            result[item] = dict(se_native=sd/math.sqrt(n),n=int(n),source="released-observer-spread")
    return complete(ledger,wanted_rows.values(),result)
