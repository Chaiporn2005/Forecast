"""Prepare development-only parameter lock; validate a supplied independent cohort.

No external patient data are bundled. CSV inputs must already be harmonized to
the development assay using a documented, outcome-blind laboratory procedure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import run_comparison as base
from run_repeated import weighted_ap

ROOT=Path(__file__).resolve().parent/'External_validation'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def development_hashes():
    return {p.name:digest(p) for p in [Path(base.__file__),base.DATA/'s_MTBLS242.txt',base.DATA/'m_MTBLS242_v2_maf.tsv']}

def prepare():
    ROOT.mkdir(exist_ok=True)
    lockpath=ROOT/'development_lock.json'
    if lockpath.exists():raise ValueError('Development lock exists; preserve the preregistered protocol.')
    pairs,x,y,names,directions,_=base.load_data()
    bests={};rows=[]
    with threadpool_limits(limits=1):
        for name,grid in base.GRIDS.items():
            means=[]
            for candidate,params in enumerate(grid):
                aps=[]
                for fold,valid in enumerate(base.splits(y,5,90210),1):
                    train=np.setdiff1d(np.arange(len(y)),valid)
                    score,_,_=base.fit_score(name,params,x[train],y[train],x[valid],directions)
                    ap=average_precision_score(y[valid],score);aps.append(ap)
                    rows.append(dict(model=name,candidate=candidate,fold=fold,ap=ap,parameters=json.dumps(params)))
                means.append(float(np.mean(aps)))
            winner=int(np.argmax(means))
            bests[name]=dict(parameters=grid[winner],development_cv_ap=means[winner])
    chosen=max(bests,key=lambda n:bests[n]['development_cv_ap'])
    lock=dict(status='READY_FOR_INDEPENDENT_DATA_NOT_EXTERNALLY_VALIDATED',development_cohort='MTBLS242',
              sha256=development_hashes(),features=names,models=bests,primary_model=chosen,
              selection='fixed development-only stratified 5-fold seed 90210; refit on all 71 development patients',
              target='>=9 of 11 prespecified markers improve at 12 months',n_subjects=71,
              sklearn_version=base.sklearn.__version__)
    lockpath.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(ROOT/'development_tuning.csv',index=False)
    pd.DataFrame(columns=['subject_id']+[f'preop::{n}' for n in names]+[f'month12::{n}' for n in names if n in base.SIGNS]).to_csv(ROOT/'external_input_template.csv',index=False)
    provenance=dict(cohort_id='',source_url='',independence_evidence='',serum_assay_harmonization='',
                    feature_identity_and_units_review='',preop_and_12month_pairing_review='',
                    outcome_blind_processing_evidence='',data_use_permission='')
    (ROOT/'provenance_template.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(lock,indent=2))

def validate(csv_path, provenance_path):
    lock=json.loads((ROOT/'development_lock.json').read_text())
    assert lock['sha256']==development_hashes(),'Development inputs or model implementation changed.'
    assert lock['sklearn_version']==base.sklearn.__version__
    provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    template=json.loads((ROOT/'provenance_template.json').read_text())
    for key in template:
        assert isinstance(provenance.get(key),str) and provenance[key].strip(),f'Missing evidence: {key}'
    assert 'MTBLS242' not in provenance['cohort_id'].upper(),'Development cohort cannot validate itself.'
    data=pd.read_csv(csv_path,dtype={'subject_id':str})
    pairs,x,y,names,directions,_=base.load_data()
    assert len(data)>0 and data.subject_id.notna().all() and data.subject_id.is_unique
    assert not set(data.subject_id)&set(pairs.subject_id),'Subject IDs overlap; independence requires review.'
    pre=data[[f'preop::{n}' for n in names]].to_numpy(float)
    marker_names=[n for n in names if n in base.SIGNS]
    post=data[[f'month12::{n}' for n in marker_names]].to_numpy(float)
    assert np.isfinite(pre).all() and np.isfinite(post).all() and (pre>=0).all() and (post>=0).all()
    # Model fitting never sees external outcomes or recalibrates on this cohort.
    pred={}
    with threadpool_limits(limits=1):
        for name,detail in lock['models'].items():
            pred[name],_,_=base.fit_score(name,detail['parameters'],x,y,np.log1p(pre),directions)
    change=post-pre[:,[names.index(n) for n in marker_names]]
    count=(change*np.array([base.SIGNS[n] for n in marker_names])<0).sum(axis=1)
    target=(count>=9).astype(int)
    assert len(np.unique(target))==2,'Both outcome classes are required to assess discrimination.'
    rng=np.random.default_rng(90211)
    weights=rng.multinomial(len(target),np.full(len(target),1/len(target)),size=2000).astype(float)
    valid=(weights@target>0)&(weights@(1-target)>0);weights=weights[valid]
    rows=[]
    for name,scores in pred.items():
        lo,hi=np.quantile(weighted_ap(target,scores,weights),[.025,.975])
        rows.append(dict(model=name,primary=name==lock['primary_model'],n=len(target),n_positive=int(target.sum()),
                         ap=average_precision_score(target,scores),ci95_low=lo,ci95_high=hi,prevalence=target.mean()))
    output=ROOT/'evaluated'
    if output.exists():raise ValueError('External results exist. Repeated external peeking requires a new explicit protocol.')
    output.mkdir()
    result=data[['subject_id']].copy();result['target']=target;result['improved_count']=count
    for name,scores in pred.items():result[name+'_score']=scores
    result.to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    (output/'provenance.json').write_text(json.dumps(dict(evidence=provenance,input_sha256=digest(csv_path),
        lock_sha256=digest(ROOT/'development_lock.json'),bootstrap_valid_draws=len(weights),
        clinical_efficacy_established=False),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--data',type=Path);parser.add_argument('--provenance',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.data and args.provenance:validate(args.data,args.provenance)
    else:parser.error('Use --prepare or --data paired.csv --provenance evidence.json')
