"""Software guard tests only. Generates no external cohort or performance claim."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import external_validation as ev

def main():
    template=json.loads((ev.ROOT/'provenance_template.json').read_text())
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'test_evidence.json'
        path.write_text(json.dumps(template))
        try:ev.validate(Path(temp)/'missing.csv',path)
        except AssertionError as error:assert 'Missing evidence' in str(error)
        else:raise AssertionError('Empty provenance accepted')
        evidence={k:'UNIT TEST ONLY' for k in template}
        evidence['cohort_id']='MTBLS242';path.write_text(json.dumps(evidence))
        try:ev.validate(Path(temp)/'missing.csv',path)
        except AssertionError as error:assert 'cannot validate itself' in str(error)
        else:raise AssertionError('Development cohort accepted')
        evidence['cohort_id']='TEST-INDEPENDENT';path.write_text(json.dumps(evidence))
        development=ev.base.load_data()
        with patch.object(ev.base,'load_data',return_value=development), patch.object(ev.pd,'read_csv',return_value=pd.DataFrame({'subject_id':[development[0].subject_id.iloc[0]]})):
            try:ev.validate(Path(temp)/'missing.csv',path)
            except AssertionError as error:assert 'overlap' in str(error)
            else:raise AssertionError('Overlapping subject accepted')
    report=dict(status='PASS',checks=['empty provenance rejected','MTBLS242 rejected as its own external cohort',
        'overlapping subject IDs rejected'],performance_evaluation_performed=False)
    (ev.ROOT/'guard_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
