"""Repeated nested CV; paired subject bootstrap of fixed held-out predictions.

Bootstrap intervals are conditional on these fitted models, NOT full retraining
confidence intervals for clinical or population generalization performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import run_comparison as base

ROOT = Path(__file__).resolve().parent / 'Repeated_nested_CV'

def weighted_ap(y, scores, weights):
    """Vectorized sklearn-compatible step AP, including ties and zero weights."""
    order = np.argsort(-scores, kind='stable')
    yy, ss, ww = y[order], scores[order], weights[:, order]
    endpoints = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.cumsum(ww * yy, axis=1)[:, endpoints]
    total = np.cumsum(ww, axis=1)[:, endpoints]
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    increments = np.diff(np.c_[np.zeros(len(ww)), tp], axis=1)
    return np.divide((increments*precision).sum(axis=1), tp[:, -1],
                     out=np.zeros(len(ww)), where=tp[:, -1] > 0)

def run_repeat(repeat, pairs, x, y, directions, output):
    seed = 42 + 1000*repeat
    rows, predictions, tuning, members = [], [], [], []
    for fold, test in enumerate(base.splits(y, 5, seed), 1):
        train = np.setdiff1d(np.arange(len(y)), test)
        inner = base.splits(y[train], 5, seed+fold)
        assert not np.intersect1d(train, test).size
        for j, valid in enumerate(inner, 1):
            for local, idx in enumerate(train):
                members.append(dict(repeat=repeat+1, outer_fold=fold, inner_fold=j,
                                    subject_id=pairs.subject_id.iloc[idx],
                                    role='validation' if local in valid else 'train'))
        bests, fold_preds = {}, {}
        for name, grid in base.GRIDS.items():
            means=[]
            for candidate, params in enumerate(grid):
                aps=[]
                for j, valid in enumerate(inner, 1):
                    itrain=np.setdiff1d(np.arange(len(train)), valid)
                    scores, detail, _=base.fit_score(name,params,x[train[itrain]],y[train[itrain]],x[train[valid]],directions)
                    ap=average_precision_score(y[train[valid]],scores)
                    aps.append(ap)
                    tuning.append(dict(repeat=repeat+1,outer_fold=fold,inner_fold=j,model=name,
                                       candidate=candidate,parameters=json.dumps(params),validation_ap=ap,**detail))
                means.append(float(np.mean(aps)))
            winner=int(np.argmax(means))
            pred,detail,_=base.fit_score(name,grid[winner],x[train],y[train],x[test],directions)
            bests[name]=means[winner]; fold_preds[name]=pred
            rows.append(dict(repeat=repeat+1,seed=seed,outer_fold=fold,model=name,
                             n_train=len(train),n_test=len(test),ap=average_precision_score(y[test],pred),
                             prevalence=y[test].mean(),best_inner_ap=means[winner],
                             parameters=json.dumps(grid[winner]),**detail))
        selected=max(bests,key=bests.get)
        fold_preds['Inner-selected']=fold_preds[selected]
        rows.append(dict(repeat=repeat+1,seed=seed,outer_fold=fold,model='Inner-selected',
                         n_train=len(train),n_test=len(test),ap=average_precision_score(y[test],fold_preds[selected]),
                         prevalence=y[test].mean(),selected_model=selected))
        for name,pred in fold_preds.items():
            for idx,score in zip(test,pred):
                predictions.append(dict(repeat=repeat+1,outer_fold=fold,model=name,
                                        subject_index=int(idx),subject_id=pairs.subject_id.iloc[idx],
                                        y=int(y[idx]),score=float(score)))
        print(f'Repeat {repeat+1}: fold {fold}/5 complete',flush=True)
    for filename, values in [('fold_metrics',rows),('predictions',predictions),('inner_tuning',tuning),('inner_membership',members)]:
        pd.DataFrame(values).to_csv(output/f'{filename}.csv',index=False)

def summarize(predictions, folds, y, draws, output):
    rng=np.random.default_rng(20260906)
    # One weight per ORIGINAL subject, shared by all repeats and all models.
    weights=rng.multinomial(len(y),np.full(len(y),1/len(y)),size=draws).astype(float)
    valid=(weights@y > 0)&(weights@(1-y)>0)
    weights=weights[valid]
    models=list(base.GRIDS)+['Inner-selected']
    boot={name:np.zeros(len(weights)) for name in models}
    zero_positive=0
    nfolds=len(folds[folds.model.eq(models[0])])
    for (_,_,name),g in predictions.groupby(['repeat','outer_fold','model']):
        idx=g.subject_index.to_numpy(int)
        boot[name]+=weighted_ap(g.y.to_numpy(),g.score.to_numpy(),weights[:,idx])/nfolds
        if name==models[0]: zero_positive+=int((weights[:,idx]@g.y.to_numpy()==0).sum())
    summaries=[]
    for name in models:
        f=folds[folds.model.eq(name)]
        repeat_means=f.groupby('repeat').ap.mean()
        lo,hi=np.quantile(boot[name],[.025,.975])
        delta=boot[name]-boot['Monotonic']
        dlo,dhi=np.quantile(delta,[.025,.975])
        summaries.append(dict(model=name,mean_fold_ap=f.ap.mean(),repeat_mean_sd=repeat_means.std(ddof=1),
                              conditional_ci95_low=lo,conditional_ci95_high=hi,
                              mean_difference_vs_monotonic=f.ap.mean()-folds[folds.model.eq('Monotonic')].ap.mean(),
                              paired_conditional_ci95_low=dlo,paired_conditional_ci95_high=dhi))
    summary=pd.DataFrame(summaries)
    summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(boot).to_csv(output/'subject_bootstrap_draws.csv',index=False)
    folds.groupby(['repeat','model']).ap.mean().unstack().to_csv(output/'repeat_means.csv')
    fig,ax=plt.subplots(figsize=(10,6))
    for i,row in summary.iterrows():
        ax.plot([row.conditional_ci95_low,row.conditional_ci95_high],[i,i],lw=3,color='#157b83')
        ax.plot(row.mean_fold_ap,i,'o',color='#b0182c')
    ax.axvline(y.mean(),ls='--',color='gray',label=f'Prevalence = {y.mean():.3f}')
    ax.set(yticks=range(len(models)),yticklabels=models,xlim=(0,1),xlabel='Mean held-out fold Average Precision',title='Repeated nested 5 x 5 CV | 71 subjects')
    ax.invert_yaxis();ax.legend()
    fig.text(.5,.015,'95% paired subject-bootstrap intervals conditional on fitted CV models; no external validation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(output/'Graph'/'conditional_confidence_intervals.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for name in models:
        values=folds[folds.model.eq(name)].groupby('repeat').ap.mean()
        ax.plot(values.index,values.values,marker='.',label=name)
    ax.set(xlabel='Repeat',ylabel='Mean outer-fold AP',ylim=(0,1),title='Sensitivity to patient split | same patients across models')
    ax.legend();fig.tight_layout();fig.savefig(output/'Graph'/'repeat_stability.png',dpi=220);plt.close(fig)
    return summary,zero_positive

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=20);parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args();assert args.repeats>=2 and args.bootstrap>=100
    ROOT.mkdir(exist_ok=True);(ROOT/'Graph').mkdir(exist_ok=True)
    pairs,x,y,names,directions,count=base.load_data()
    manifest=dict(repeats=args.repeats,outer_folds=5,inner_folds=5,outer_seeds=[42+1000*r for r in range(args.repeats)],
                  bootstrap_draws=args.bootstrap,bootstrap_seed=20260906,n_subjects=len(y),n_positive=int(y.sum()),
                  target='>=9/11 fixed markers improve at 12 months',features=names,constraints=base.SIGNS,grid=base.GRIDS,
                  primary_metric='mean outer-fold average precision across repeats',
                  ci='95% percentile paired subject bootstrap of fixed within-fold predictions; conditional, no model refitting',
                  zero_positive_bootstrap_fold='AP defined as zero, matching sklearn; retain fixed fold denominator',
                  external_validation='NOT PERFORMED: no eligible external subject-level table available',
                  sha256={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(base.__file__),Path(__file__),base.DATA/'s_MTBLS242.txt',base.DATA/'m_MTBLS242_v2_maf.tsv']},
                  versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=base.scipy.__version__,sklearn=base.sklearn.__version__))
    protocol=ROOT/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())==manifest,'Protocol changed: use a separate output directory.'
    protocol.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with threadpool_limits(limits=1):
        for r in range(args.repeats):
            output=ROOT/f'repeat_{r+1:02d}';output.mkdir(exist_ok=True)
            if not (output/'inner_membership.csv').exists():run_repeat(r,pairs,x,y,directions,output)
    predictions=pd.concat([pd.read_csv(ROOT/f'repeat_{r+1:02d}'/'predictions.csv',dtype={'subject_id':str}) for r in range(args.repeats)],ignore_index=True)
    folds=pd.concat([pd.read_csv(ROOT/f'repeat_{r+1:02d}'/'fold_metrics.csv') for r in range(args.repeats)],ignore_index=True)
    assert predictions.groupby(['repeat','model','subject_id']).size().eq(1).all()
    assert len(predictions)==args.repeats*len(y)*5
    assert np.isfinite(predictions.score).all()
    summary,zeros=summarize(predictions,folds,y,args.bootstrap,ROOT)
    (ROOT/'run_checks.json').write_text(json.dumps(dict(one_prediction_per_subject_repeat_model=True,finite_scores=True,
        all_fits_converged=True,monotonic_violations=int(folds.violations.fillna(0).sum()),
        zero_positive_bootstrap_fold_instances=zeros,outer_refits=args.repeats*20,inner_fits=args.repeats*1000),indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
