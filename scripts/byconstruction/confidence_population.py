"""Freeze native uncertainty row identities from original Mlic fitting keys only.

This metadata step opens no feature/target table and parses no uncertainty.
Raw source digests are opaque integrity reads. It does not admit linking units.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
from subjective_training_leg import document, safe, sha
from confidence_sources import digest
from stimulus_identity import promoted_stimuli, resolve

ROOT_SHA = "a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952"
ROOT = Path("/var/tmp/byc-mixes-20261025/Mlic/root")
LIBRARY = Path("/mnt/tower/input/iqa/datasets")
MEMBERS = dict(kadid_train="kadid-10k",kadid_select="kadid-10k",tid2013="tid2013")


def identities(path):
    """The normalized schema puts its two image identifiers before any response."""
    result = {}
    with safe(path).open() as stream:
        header = next(csv.reader([next(stream)],delimiter="\t"))
        if header[:2] != ["item","ref"]:
            raise ValueError("native identifier prefix changed")
        for i,line in enumerate(stream):
            prefix = line.split("\t",2)
            if len(prefix) < 3:
                raise ValueError("native identifier row shape differs")
            item,ref = next(csv.reader(["\t".join(prefix[:2])],delimiter="\t"))
            if item in result:
                raise ValueError("ambiguous native item identity")
            result[item] = dict(row=i,item=item,reference=ref)
    return result


def bind_human(keys,banks,native,sources):
    by_name = {}
    for dataset in sorted(set(MEMBERS.values())):
        rows = native[dataset]
        by_name[dataset] = {}
        for row in rows.values():
            key = Path(row["item"]).stem.lower()
            if key in by_name[dataset]:
                raise ValueError("ambiguous native stimulus stem")
            by_name[dataset][key] = row
    result=[]
    seen=set()
    for index,key in enumerate(keys):
        member = key["member_set"]
        if member not in MEMBERS:
            continue
        identity=(member,key["pair_key"],key["source_row_id"])
        if identity in seen:
            raise ValueError("duplicate fitting stimulus identity")
        seen.add(identity)
        bank = banks[member][key["pair_key"]]
        source = resolve(key,banks,sources)
        dataset = MEMBERS[member]
        native_row = by_name[dataset][Path(source["dist_path"]).stem.lower()]
        if Path(native_row["reference"]).stem.lower() != Path(source["ref_path"]).stem.lower():
            raise ValueError("native/promoted reference identity differs")
        result.append(dict(table_row=index,pair_key=key["pair_key"],source_row_id=key["source_row_id"],member_set=member,collection=dataset,
            reference_group=key["ref_basename"],native=native_row,ref_pixels_sha256=bank["ref_pixels_sha256"],
            dist_pixels_sha256=bank["dist_pixels_sha256"]))
    return result


def prepare(output):
    frozen=document(ROOT/"wide/frozen.json",ROOT_SHA)
    output.mkdir()  # Fresh metadata population only.
    receipt=document(ROOT/"wide/main/real/receipt.json",frozen["wide_receipts"]["main/real"])
    inputs={str(ROOT/"wide/frozen.json"):ROOT_SHA,
            str(ROOT/"wide/main/real/receipt.json"):sha(ROOT/"wide/main/real/receipt.json")}
    native={}
    for dataset in (*sorted(set(MEMBERS.values())),"live-md","colourlab-rating-variability-2022"):
        p=LIBRARY/dataset/"labels.tsv"
        inputs[str(p)]=digest(p)
        native[dataset]=identities(p)
    banks,sources,source_inputs=promoted_stimuli(MEMBERS)
    inputs.update(source_inputs)
    tables={}
    for name,leg in receipt["legs"].items():
        if name!="human_all" and not name.startswith("human_without_"): continue
        fit=leg["fit"];table=ROOT/fit["rel"]
        manifest=Path(str(table)+".manifest.json")
        d=document(manifest,fit["manifest_sha256"])
        kp=table.with_suffix(".keys.parquet")
        if sha(kp)!=fit["keys_sha256"] or d["keys_sha256"]!=fit["keys_sha256"]:
            raise ValueError("fitting row key pin changed")
        inputs[str(kp)]=fit["keys_sha256"];inputs[str(manifest)]=fit["manifest_sha256"]
        keys=pq.read_table(kp).to_pylist()
        tables[name]=dict(table=str(table),table_sha256=fit["sha256"],keys_sha256=fit["keys_sha256"],
            rows=len(keys),selection=bind_human(keys,banks,native,sources))
    packet=document(ROOT/"datasets/receipt.json",frozen["auxiliary_files"]["datasets/receipt.json"])
    inputs[str(ROOT/"datasets/receipt.json")]=sha(ROOT/"datasets/receipt.json")
    for group in packet["groups"]:
        dataset=group["collection"]
        if dataset not in ("live-md","colourlab-rating-variability-2022"):continue
        keys=document(group["keys_path"],group["keys_sha256"])
        # The canonical native source row is bound by the original leg keys.
        rows=keys["rows"] if isinstance(keys,dict) else keys
        if len(rows)!=group["rows"]:raise ValueError("native fitting population differs")
        by_row={r["row"]:r for r in native[dataset].values()}
        selected=[]
        for i,key in enumerate(rows):
            collection,protocol,index=key["observation"].split(":")
            native_row=by_row[int(index)]
            if (collection!=dataset or protocol!="labels.tsv" or key["role"]!="fit" or key["row"]!=i
                    or key["reference"]!=f"{dataset}:{native_row['reference']}"):
                raise ValueError("native fitting source binding differs")
            selected.append(dict(table_row=i,observation=key["observation"],collection=dataset,
                                 native=native_row,family=key["family"]))
        inputs[group["keys_path"]]=group["keys_sha256"]
        tables[group["name"]]=dict(table=group["path"],table_sha256=group["table_sha256"],
            keys_sha256=group["keys_sha256"],rows=group["rows"],selection=selected)
    raw={}
    for dataset,files in {"kadid-10k":["labels/dmos.csv"],"tid2013":["labels/mos_with_names.txt","labels/mos_std.txt"],
            "live-md":[f"labels/Part {p}/{f}.mat" for p in (1,2) for f in ("Imagelists","Scores")]}.items():
        raw[dataset]={str(LIBRARY/dataset/f):digest(LIBRARY/dataset/f) for f in files}
    value=dict(schema="mlic-native-uncertainty-populations-v1",role="TRAIN",original_root_sha256=ROOT_SHA,
        response_values_parsed=False,uncertainty_values_parsed=False,input_sha256=inputs,raw_sources=raw,tables=tables,
        live_md_state="observer-fit population and source-column admission required",linking_state="fresh TRAIN admission required")
    path=output/"POPULATIONS.json";path.write_text(json.dumps(value,indent=1)+"\n")
    print(json.dumps(dict(path=str(path),sha256=sha(path),tables={k:len(v["selection"]) for k,v in tables.items()})))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    prepare(parser.parse_args().output)
