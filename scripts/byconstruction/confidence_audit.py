"""Audit native TRAIN uncertainty; no weighting, calibration or candidate fitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

from confidence_sources import authorize, complete, digest, kadid, tid, published_spread
from confidence_population import ROOT_SHA, LIBRARY
from live_md_uncertainty import selected_columns, sureal

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
from subjective_training_leg import document, sha


def live_md(population,pins,ledger):
    parts=[]
    for part,observers in (("Part 1",19),("Part 2",18)):
        wanted={r["native"]["item"] for r in population if f"/{part}/" in r["native"]["item"]}
        names_path=LIBRARY/"live-md/labels"/part/"Imagelists.mat"
        authorize(ledger,names_path,pins[str(names_path)],wanted,["distimgs"],"native observer-column identities")
        names=[str(x[0][0]).lower() for x in loadmat(names_path,variable_names=["distimgs"])["distimgs"]]
        if len(names)!=225 or len(set(names))!=225:raise ValueError("LIVE-MD native name population differs")
        lookup={Path(item).name.lower():item for item in wanted}
        indices=[i for i,name in enumerate(names) if name in lookup]
        items=[lookup[names[i]] for i in indices]
        if len(items)!=len(wanted):raise ValueError("LIVE-MD fitting item join incomplete")
        complete(ledger,wanted,{item:True for item in items})
        scores=names_path.with_name("Scores.mat")
        authorize(ledger,scores,pins[str(scores)],wanted,
            [dict(variable="dist_diffscores",observer_rows=list(range(observers)),item_columns=indices)],
            "selected native observer differences; reference minus test")
        values=selected_columns(scores.read_bytes(),"dist_diffscores",indices,(observers,225))
        complete(ledger,wanted,{item:True for item in items})
        parts.append((items,values.T))
    items=[item for names,_ in parts for item in names]
    if len(items)!=len(population) or len(set(items))!=len(items):raise ValueError("LIVE-MD fitting population differs")
    R=np.full((len(items),sum(v.shape[1] for _,v in parts)),np.nan)
    row=col=0
    for names,values in parts:
        R[row:row+len(names),col:col+values.shape[1]]=values
        row+=len(names);col+=values.shape[1]
    keep=np.where((~np.isnan(R)).sum(0)>=5)[0]
    if len(keep)!=R.shape[1]:raise ValueError("native observer population requires explicit review")
    _,_,_,se=sureal(R)
    if not np.isfinite(se).all() or not (se>0).all():raise ValueError("invalid fitted observer uncertainty")
    return {item:dict(se_native=float(se[i]),n=int(np.isfinite(R[i]).sum()),
        source="zenpapers-sureal-frozen-TRAIN-observers") for i,item in enumerate(items)}


def audit(populations,pin,output):
    d=document(populations,pin)
    if (d.get("schema")!="mlic-native-uncertainty-populations-v1" or d.get("role")!="TRAIN"
            or d.get("original_root_sha256")!=ROOT_SHA or d.get("response_values_parsed") is not False
            or d.get("uncertainty_values_parsed") is not False):
        raise ValueError("frozen metadata-only fitting population required")
    output.mkdir()
    ledger=output/"LABEL_READS.json"
    ledger.write_text(json.dumps(dict(role="TRAIN",fitting_manifest_sha256=pin,reads=[]),indent=1)+"\n")
    state=dict(stage="source-integrity",source_script_sha256=sha(__file__),population_sha256=pin,
        producer_sources={name:sha(Path(__file__).with_name(name)) for name in
            ("confidence_sources.py","confidence_population.py","live_md_uncertainty.py")},
        updated_utc=datetime.now(timezone.utc).isoformat(),weights_constructed=False,candidate_model_fit=False)
    state["producer_sources"]["stimulus_identity.py"]=sha(Path(__file__).parents[1]/"rev4_featpot/stimulus_identity.py")
    def record():
        (output/"STATE.json").write_text(json.dumps(state,indent=1)+"\n")
    record()
    try:
        # Recheck only metadata/key/native-source digests, never feature tables.
        reads=json.loads(ledger.read_text())
        reads["reads"].append(dict(kind="opaque-population-input-integrity",input_sha256=d["input_sha256"]))
        ledger.write_text(json.dumps(reads,indent=1)+"\n")
        for path,expected in d["input_sha256"].items():
            if digest(path)!=expected:raise ValueError("frozen population metadata/source changed")
        raw=d["raw_sources"]
        human=d["tables"]["human_all"]["selection"]
        for collection in ("kadid-10k","tid2013"):
            selected=[r for r in human if r["collection"]==collection]
            if (any("source_row_id" not in r for r in selected)
                    or len({r["native"]["item"] for r in selected})!=len(selected)):
                raise ValueError("native uncertainty must preserve every fitting stimulus")
        result={}
        state["stage"]="kadid-10k";record()
        wanted={Path(r["native"]["item"]).name.lower() for r in human if r["collection"]=="kadid-10k"}
        path=LIBRARY/"kadid-10k/labels/dmos.csv"
        result["kadid-10k"]=kadid(path,raw["kadid-10k"][str(path)],wanted,ledger)
        state["stage"]="tid2013";record()
        wanted={Path(r["native"]["item"]).name.lower() for r in human if r["collection"]=="tid2013"}
        names=LIBRARY/"tid2013/labels/mos_with_names.txt";std=names.with_name("mos_std.txt")
        result["tid2013"]=tid(names,raw["tid2013"][str(names)],std,raw["tid2013"][str(std)],wanted,ledger)
        state["stage"]="colourlab-rating-variability-2022";record()
        selection=d["tables"]["subjective_rank_colourlab_rating_variability_2022"]["selection"]
        path=LIBRARY/"colourlab-rating-variability-2022/labels.tsv"
        result["colourlab-rating-variability-2022"]=published_spread(path,d["input_sha256"][str(path)],
            {r["native"]["row"]:r["native"]["item"] for r in selection},ledger)
        state["stage"]="live-md-observer-fit";record()
        result["live-md"]=live_md(d["tables"]["subjective_rank_live_md"]["selection"],raw["live-md"],ledger)
        path=output/"NATIVE_UNCERTAINTY.json"
        path.write_text(json.dumps(dict(schema="mlic-native-uncertainty-audit-v1",role="TRAIN",population_sha256=pin,
            datasets=result,linking_admitted=False,weights_constructed=False),indent=1)+"\n")
        state.update(stage="native-uncertainty-verified",datasets={k:len(v) for k,v in result.items()},
            output_sha256=sha(path),updated_utc=datetime.now(timezone.utc).isoformat())
        record();print(json.dumps(state))
    except BaseException as error:
        state.update(failure=repr(error),updated_utc=datetime.now(timezone.utc).isoformat());record();raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("populations",type=Path);p.add_argument("sha256");p.add_argument("output",type=Path)
    a=p.parse_args();audit(a.populations,a.sha256,a.output)
