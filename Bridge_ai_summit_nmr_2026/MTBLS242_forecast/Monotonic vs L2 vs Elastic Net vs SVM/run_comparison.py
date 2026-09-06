"""Reproduce the fixed-target, paired nested 5 x 5 CV comparison."""
from pathlib import Path
import hashlib
import json
import platform
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
SEED = 42
SIGNS = {'L-valine':1, 'L-leucine':1, 'L-allo-Isoleucine':1,
         'L-tyrosine':1, 'D-phenylalanine':1, 'L-alanine':1,
         'lipoproteins':1, 'L-Lactic acid':1, 'glycine':-1,
         'L-glutamine':-1, 'histidine':-1}
LAMBDAS = [0.001, 0.01, 0.1, 1., 10.]
GRIDS = {
 'Monotonic':[{'lambda':v} for v in LAMBDAS],
 'L2':[{'lambda':v} for v in LAMBDAS],
 'Elastic Net':[{'lambda':v,'l1_ratio':r} for v in LAMBDAS for r in [0.1,0.5,0.9]],
 'SVM':[{'C':c,'gamma':g} for c in [0.01,0.1,1.,10.,100.] for g in ['scale',0.01,0.1]],
}

def splits(y, k, seed):
    # Exactly the legacy outer allocation; shared across every model.
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for i, part in enumerate(np.array_split(idx, k)):
            parts[i].append(part)
    return [np.sort(np.concatenate(p)) for p in parts]

def load_data():
    sample = pd.read_csv(DATA/'s_MTBLS242.txt', sep='\t', dtype=str)
    sample['subject_id'] = sample['Sample Name'].str.extract(r'(?:^|[_-])(\d{4})(?:[_-]|$)')[0]
    assert sample.subject_id.notna().all()
    pre = sample[sample['Factor Value[time point]'].eq('preop')]
    post = sample[sample['Factor Value[time point]'].eq('12 months after surgery')]
    pairs = pre[['subject_id','Sample Name']].merge(post[['subject_id','Sample Name']],on='subject_id',suffixes=('_preop','_12mo')).sort_values('subject_id').reset_index(drop=True)
    assert pairs.subject_id.is_unique
    maf = pd.read_csv(DATA/'m_MTBLS242_v2_maf.tsv',sep='\t')
    names = maf.metabolite_identification.tolist()
    assert len(names) == len(set(names)) == 21
    table = maf.set_index('metabolite_identification')
    x = table[pairs['Sample Name_preop']].T.to_numpy(float)
    future = table[pairs['Sample Name_12mo']].T.to_numpy(float)
    # Fail closed: no full-cohort feature filtering, selection, or imputation.
    assert np.isfinite(x).all() and np.isfinite(future).all()
    assert (x >= 0).all() and (future >= 0).all()
    x, future = np.log1p(x), np.log1p(future)
    directions = np.array([SIGNS.get(n,0) for n in names])
    count = ((future-x)*directions < 0).sum(axis=1)
    y = (count >= 9).astype(int)
    assert len(y)==71 and y.sum()==31
    legacy = pd.read_csv(DATA/'outputs_pr_auc'/'heldout_responder_predictions.csv',dtype={'subject_id':str})
    assert legacy.subject_id.tolist() == pairs.subject_id.tolist()
    assert np.array_equal(legacy.strong_responder,y)
    assert np.array_equal(legacy.improved_marker_count,count)
    return pairs,x,y,names,directions,count

def fit_score(name, params, x, y, test, directions):
    scaler = StandardScaler().fit(x)
    z, zz = scaler.transform(x),scaler.transform(test)
    detail = {'iterations':0, 'nonzero_features':len(directions), 'violations':None}
    if name in ['Monotonic','L2']:
        lam=params['lambda']
        def loss(w):
            score=z@w[:-1]+w[-1]
            err=expit(score)-y
            return (np.mean(np.logaddexp(0,score)-y*score)+lam/2*np.sum(w[:-1]**2),
                    np.r_[z.T@err/len(y)+lam*w[:-1],err.mean()])
        bounds=[(0,None) if d==1 else (None,0) if d==-1 else (None,None) for d in directions] if name=='Monotonic' else [(None,None)]*len(directions)
        initial=np.r_[np.zeros(z.shape[1]),np.log(y.mean()/(1-y.mean()))]
        result=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=bounds+[(None,None)],options={'maxiter':10000,'ftol':1e-12,'gtol':1e-8})
        if not result.success: raise RuntimeError(result.message)
        coef=result.x[:-1]
        score=expit(zz@coef+result.x[-1])
        detail.update(iterations=int(result.nit),nonzero_features=int((abs(coef)>1e-8).sum()))
        if name=='Monotonic':
            detail['violations']=int(((directions!=0)&(directions*coef < -1e-10)).sum())
            assert detail['violations']==0
    elif name=='Elastic Net':
        # sklearn sum-loss convention: C = 1 / (n * lambda).
        model=LogisticRegression(solver='saga',l1_ratio=params['l1_ratio'],C=1/(len(y)*params['lambda']),max_iter=50000,tol=1e-7,random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(z,y)
        coef=model.coef_[0]
        score=model.predict_proba(zz)[:,1]
        detail.update(iterations=int(model.n_iter_[0]),nonzero_features=int((abs(coef)>1e-8).sum()))
    else:
        model=SVC(kernel='rbf',C=params['C'],gamma=params['gamma'])
        model.fit(z,y)
        # Raw margins rank cases within a fold; not calibrated probabilities.
        score=model.decision_function(zz)
        coef=None
        detail['iterations']=int(model.n_iter_[0])
    return score,detail,coef

def main():
    results=ROOT/'Results'; graph=ROOT/'Graph'
    results.mkdir(exist_ok=True); graph.mkdir(exist_ok=True)
    pairs,x,y,names,directions,count=load_data()
    outer=splits(y,5,SEED)
    oof={name:np.full(len(y),np.nan) for name in GRIDS}
    assignments=np.zeros(len(y),int)
    fold_rows=[]; tuning=[]; memberships=[]; coef_rows=[]; selected=np.full(len(y),np.nan); selected_rows=[]
    for fold,test in enumerate(outer,1):
        train=np.setdiff1d(np.arange(len(y)),test)
        assert not set(train)&set(test)
        assignments[test]=fold
        inner=splits(y[train],5,SEED+fold)
        for j,valid in enumerate(inner,1):
            itrain=np.setdiff1d(np.arange(len(train)),valid)
            assert not set(train[valid])&set(test)
            for local in range(len(train)):
                memberships.append({'outer_fold':fold,'inner_fold':j,'subject_id':pairs.subject_id.iloc[train[local]],'role':'validation' if local in valid else 'train'})
        fold_best={}
        for name,grid in GRIDS.items():
            candidate_means=[]
            for candidate,params in enumerate(grid):
                scores=[]
                for j,valid in enumerate(inner,1):
                    itrain=np.setdiff1d(np.arange(len(train)),valid)
                    pred,detail,_=fit_score(name,params,x[train[itrain]],y[train[itrain]],x[train[valid]],directions)
                    ap=average_precision_score(y[train[valid]],pred)
                    scores.append(ap)
                    tuning.append({'outer_fold':fold,'model':name,'candidate':candidate,'parameters':json.dumps(params),'inner_fold':j,'validation_ap':ap,**detail})
                candidate_means.append(float(np.mean(scores)))
            # Deterministic tie breaking: first predeclared grid entry.
            best=int(np.argmax(candidate_means)); params=grid[best]
            pred,detail,coef=fit_score(name,params,x[train],y[train],x[test],directions)
            oof[name][test]=pred
            row={'outer_fold':fold,'model':name,'n_train':len(train),'n_test':len(test),'test_positives':int(y[test].sum()),'baseline_prevalence':float(y[test].mean()),'pr_auc_ap':average_precision_score(y[test],pred),'best_inner_ap':candidate_means[best],'parameters':json.dumps(params),**detail}
            fold_rows.append(row); fold_best[name]=(candidate_means[best],row,pred)
            if coef is not None:
                for feature,c,d in zip(names,coef,directions):
                    coef_rows.append({'outer_fold':fold,'model':name,'metabolite':feature,'standardized_coefficient':c,'constraint':int(d) if name=='Monotonic' else 0})
            print(f"Fold {fold} {name}: AP={row['pr_auc_ap']:.4f}, params={params}",flush=True)
        chosen=max(fold_best,key=lambda n:fold_best[n][0])
        selected[test]=fold_best[chosen][2]
        selected_rows.append({'outer_fold':fold,'selected_model':chosen,'pr_auc_ap':fold_best[chosen][1]['pr_auc_ap']})
    assert all(np.isfinite(v).all() for v in oof.values())
    assert np.array_equal(np.sort(np.concatenate(outer)),np.arange(len(y)))
    folds=pd.DataFrame(fold_rows)
    summary=[]
    ref=folds[folds.model.eq('Monotonic')].pr_auc_ap.to_numpy()
    for name in GRIDS:
        a=folds[folds.model.eq(name)].pr_auc_ap.to_numpy()
        summary.append({'model':name,'mean_fold_ap':float(a.mean()),'fold_sd_population':float(a.std(ddof=0)),'pooled_oof_ap_descriptive':average_precision_score(y,oof[name]),'mean_paired_difference_vs_monotonic':float((a-ref).mean()),'folds_better_than_monotonic':int((a>ref).sum())})
    summary=pd.DataFrame(summary).sort_values('mean_fold_ap',ascending=False)
    summary.to_csv(results/'model_summary.csv',index=False)
    folds.to_csv(results/'outer_fold_metrics.csv',index=False)
    pd.DataFrame(tuning).to_csv(results/'inner_tuning_scores.csv',index=False)
    pd.DataFrame(memberships).to_csv(results/'inner_fold_membership.csv',index=False)
    pd.DataFrame(coef_rows).to_csv(results/'outer_fit_coefficients.csv',index=False)
    pd.DataFrame(selected_rows).to_csv(results/'nested_model_selection.csv',index=False)
    predictions=pairs.copy(); predictions['outer_fold']=assignments; predictions['strong_responder']=y; predictions['improved_marker_count']=count
    for name,p in oof.items(): predictions[name+'_score']=p
    predictions.to_csv(results/'heldout_predictions.csv',index=False)
    manifest={'seed':SEED,'outer_folds':5,'inner_folds':5,'n_subjects':len(y),'n_positive':int(y.sum()),'n_features':len(names),'target':'at least 9 of 11 specified markers improve at 12 months','metric':'average precision (step integral, tie-aware)','primary':'mean outer-fold AP','grid':GRIDS,'feature_selection':'Elastic Net embedded shrinkage only, inside inner training fits; all 21 fixed features otherwise','constraints':SIGNS,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__},'sha256':{f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['s_MTBLS242.txt','m_MTBLS242_v2_maf.tsv']},'checks':{'legacy_target_identical':True,'unique_subjects':True,'one_outer_prediction_each':True,'no_missing_raw_values':True,'all_monotonic_fits_sign_valid':True,'all_solvers_converged':True},'nested_model_selection_mean_ap':float(np.mean([r['pr_auc_ap'] for r in selected_rows]))}
    (results/'protocol_and_checks.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    colors=['#137f86','#dd853f','#7756a5','#417fba']
    fig,ax=plt.subplots(figsize=(10,6))
    for i,name in enumerate(GRIDS):
        vals=folds[folds.model.eq(name)].pr_auc_ap.to_numpy()
        ax.scatter(np.full(5,i)+np.linspace(-.1,.1,5),vals,color=colors[i],s=45)
        ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=0),fmt='D',color='black',capsize=7)
        ax.text(i,1.04,f'{vals.mean():.3f} ± {vals.std(ddof=0):.3f}',ha='center')
    ax.axhline(y.mean(),ls='--',color='gray',label=f'Overall prevalence = {y.mean():.3f}')
    ax.set(xticks=range(4),xticklabels=list(GRIDS),ylim=(0,1.1),ylabel='PR-AUC (Average Precision)',title='MTBLS242 | Nested 5 × 5 CV comparison')
    ax.legend(loc='lower left'); fig.text(.5,.01,'Dots: held-out folds. Diamonds/whiskers: mean ± population SD (not a confidence interval).',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(graph/'model_comparison.png'); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8)); axes=axes.ravel()
    for fold,test in enumerate(outer,1):
        ax=axes[fold-1]
        for name,color in zip(GRIDS,colors):
            p,r,_=precision_recall_curve(y[test],oof[name][test]); ax.step(r,p,where='post',color=color,label=name)
        ax.axhline(y[test].mean(),ls='--',color='gray'); ax.set(xlim=(0,1),ylim=(0,1.05),xlabel='Recall',ylabel='Precision',title=f'Outer fold {fold} | n={len(test)}')
    axes[-1].axis('off'); handles,labels=axes[0].get_legend_handles_labels(); axes[-1].legend(handles,labels,loc='center')
    fig.suptitle('Held-out PR curves by fold | identical patients across models'); fig.tight_layout(); fig.savefig(graph/'pr_curves_by_fold.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for name,color in zip(GRIDS,colors):
        ax.plot(range(1,6),folds[folds.model.eq(name)].pr_auc_ap,marker='o',color=color,label=name)
    ax.plot(range(1,6),[y[t].mean() for t in outer],ls='--',color='gray',label='Fold prevalence')
    ax.set(xticks=range(1,6),xlabel='Outer fold',ylabel='PR-AUC (Average Precision)',ylim=(0,1),title='Paired comparison on identical held-out subjects'); ax.legend(); fig.tight_layout(); fig.savefig(graph/'paired_fold_comparison.png'); plt.close(fig)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__': main()
