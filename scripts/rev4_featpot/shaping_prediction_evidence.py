"""Bind native prediction tables to completed evaluation reports."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from shaping_completion import sha


def save(directory,report,indices,predictions):
    path=directory/'predictions.npz'
    np.savez_compressed(path,original_indices=indices,native=predictions)
    report['predictions_sha256']=sha(path)


def read(directory,cell,plan_pin,admission_pin,expected_indices):
    """Caller revalidates outer admission before invoking this response reader."""
    marker=json.loads((directory/'COMPLETE.json').read_text())
    report_path=directory/'REPORT.json';pin=sha(report_path)
    if marker['cell']!=cell['id'] or marker['report_sha256']!=pin:
        raise ValueError('completed report identity differs')
    report=json.loads(report_path.read_text())
    if (report['cell']!=cell or report['plan_sha256']!=plan_pin
        or report.get('admission_sha256')!=admission_pin or report.get('outer_results_read') is not True
        or 'outer_test' not in report or 'exposed_development' in report or 'full_fit_validation' in report):
        raise ValueError('admitted outer report scope differs')
    path=directory/'predictions.npz';prediction_pin=sha(path)
    if report.get('predictions_sha256')!=prediction_pin:
        raise ValueError('prediction table is not bound to completed report')
    with np.load(path,allow_pickle=False) as table:
        if set(table.files)!={'original_indices','native'}:raise ValueError('prediction fields differ')
        ids=table['original_indices'];values=table['native']
    expected=np.asarray(expected_indices,dtype=np.int64)
    if (ids.dtype.kind not in 'iu' or ids.ndim!=1 or not np.array_equal(ids,expected)
        or values.shape!=ids.shape or not np.isfinite(values).all()):
        raise ValueError('prediction membership, shape or finite coverage differs')
    return report,values,{str(report_path):pin,str(path):prediction_pin,
                          str(directory/'COMPLETE.json'):sha(directory/'COMPLETE.json')}
