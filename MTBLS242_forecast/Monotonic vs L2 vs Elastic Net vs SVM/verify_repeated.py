"""Independently audit metrics, split membership, tuning and bootstrap arithmetic."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import run_comparison as base
from run_repeated import ROOT, weighted_ap

def main():
    pairs,x,y,names,directions,_=base.load_data()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    rng=np.random.default_rng(1906)
    # Ties, all-tied scores, zero weights, class absence and arbitrary weights.
    for scores in [rng.integers(0,4,20),np.ones(20),rng.normal(size=20)]:
        yy=np.r_[np.zeros(10),np.ones(10)]
        w=rng.integers(0,4,(50,20)).astype(float);w[0,10:]=0;w[1,:10]=0
        actual=weighted_ap(yy,scores,w)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected=[average_precision_score(yy,scores,sample_weight=ww) for ww in w]
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    aps=[]
    for repeat in range(1,protocol['repeats']+1):
        folder=ROOT/f'repeat_{repeat:02d}'
        pred=pd.read_csv(folder/'predictions.csv',dtype={'subject_id':str})
        metrics=pd.read_csv(folder/'fold_metrics.csv')
        tuning=pd.read_csv(folder/'inner_tuning.csv')
        members=pd.read_csv(folder/'inner_membership.csv',dtype={'subject_id':str})
        seed=protocol['outer_seeds'][repeat-1]
        assert len(pred)==71*5
        assert pred.groupby(['model','subject_id']).size().eq(1).all()
        for fold,test in enumerate(base.splits(y,5,seed),1):
            train=np.setdiff1d(np.arange(len(y)),test)
            expected_test=set(pairs.subject_id.iloc[test]);expected_train=set(pairs.subject_id.iloc[train])
            for j,valid in enumerate(base.splits(y[train],5,seed+fold),1):
                m=members[(members.outer_fold==fold)&(members.inner_fold==j)]
                assert len(m)==len(train) and m.subject_id.is_unique
                assert set(m.subject_id)==expected_train
                assert set(m[m.role.eq('validation')].subject_id)==set(pairs.subject_id.iloc[train[valid]])
                assert set(m[m.role.eq('train')].subject_id)==expected_train-set(pairs.subject_id.iloc[train[valid]])
            best_inner={}
            for name in list(base.GRIDS)+['Inner-selected']:
                p=pred[(pred.outer_fold==fold)&pred.model.eq(name)]
                assert set(p.subject_id)==expected_test
                assert np.array_equal(p.y,y[p.subject_index])
                row=metrics[(metrics.outer_fold==fold)&metrics.model.eq(name)].iloc[0]
                np.testing.assert_allclose(row.ap,average_precision_score(p.y,p.score),atol=1e-12)
                if name in base.GRIDS:
                    t=tuning[(tuning.outer_fold==fold)&tuning.model.eq(name)]
                    assert len(t)==5*len(base.GRIDS[name])
                    means=t.groupby('candidate').validation_ap.mean()
                    chosen=int(np.argmax(means.to_numpy()))
                    assert json.loads(row.parameters)==base.GRIDS[name][chosen]
                    np.testing.assert_allclose(row.best_inner_ap,means.iloc[chosen],atol=1e-12)
                    best_inner[name]=means.iloc[chosen]
                else:
                    chosen=max(best_inner,key=best_inner.get)
                    assert row.selected_model==chosen
                    other=pred[(pred.outer_fold==fold)&pred.model.eq(chosen)]
                    np.testing.assert_allclose(p.sort_values('subject_id').score,other.sort_values('subject_id').score)
            aps.append(metrics[metrics.outer_fold.eq(fold)])
        if repeat==1:
            legacy=pd.read_csv(base.ROOT/'Results'/'outer_fold_metrics.csv')
            for name in base.GRIDS:
                np.testing.assert_allclose(metrics[metrics.model.eq(name)].ap,legacy[legacy.model.eq(name)].pr_auc_ap,atol=1e-12)
    all_metrics=pd.concat(aps)
    summary=pd.read_csv(ROOT/'summary.csv')
    draws=pd.read_csv(ROOT/'subject_bootstrap_draws.csv')
    for _,row in summary.iterrows():
        np.testing.assert_allclose(row.mean_fold_ap,all_metrics[all_metrics.model.eq(row.model)].ap.mean())
        np.testing.assert_allclose([row.conditional_ci95_low,row.conditional_ci95_high],draws[row.model].quantile([.025,.975]))
        np.testing.assert_allclose([row.paired_conditional_ci95_low,row.paired_conditional_ci95_high],(draws[row.model]-draws.Monotonic).quantile([.025,.975]))
    report=dict(status='PASS',repeats=protocol['repeats'],checks=['weighted AP matches sklearn including ties and zero-positive folds',
        'outer/inner patient membership reconstructed from seeds','identical outer test subjects across models',
        'each patient predicted once per repeat per model','target matches original',
        'all held-out AP recomputed','hyperparameters maximize inner mean AP','model selection uses inner scores only',
        'repeat 1 reproduces previous nested-CV results','CI quantiles recomputed from saved paired bootstrap draws'],
        external_validation='NOT PERFORMED',interval_scope='conditional on fitted CV models, no training refits')
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
