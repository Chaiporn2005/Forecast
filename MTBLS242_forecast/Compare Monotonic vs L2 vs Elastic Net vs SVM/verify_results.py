"""Audit patient separation, metric calculations, and logistic objective parity."""
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import run_comparison as c

sys.path.insert(0,str(c.DATA))
from forecast_responder_pr_auc import stratified_folds

pairs,x,y,names,directions,count=c.load_data()
res=c.ROOT/'Results'
pred=pd.read_csv(res/'heldout_predictions.csv',dtype={'subject_id':str})
folds=pd.read_csv(res/'outer_fold_metrics.csv')
tuning=pd.read_csv(res/'inner_tuning_scores.csv')
member=pd.read_csv(res/'inner_fold_membership.csv',dtype={'subject_id':str})
assert len(folds)==20 and len(pred)==71 and len(tuning)==1000
assert pred.subject_id.is_unique and pred.subject_id.tolist()==pairs.subject_id.tolist()
assert np.array_equal(pred.strong_responder,y)
assert average_precision_score([0,1,0,1],[.5,.5,.5,.5])==.5
outer=stratified_folds(y,5,42)
for i,test in enumerate(outer,1):
    assert set(pred.loc[pred.outer_fold.eq(i),'subject_id'])==set(pairs.subject_id.iloc[test])
    m=member[member.outer_fold.eq(i)]
    assert not set(m.subject_id)&set(pairs.subject_id.iloc[test])
    for j,part in m.groupby('inner_fold'):
        assert part.subject_id.is_unique
        assert set(part.subject_id)==set(pairs.subject_id)-set(pairs.subject_id.iloc[test])
    val_counts=m[m.role.eq('validation')].subject_id.value_counts()
    assert val_counts.eq(1).all() and len(val_counts)==71-len(test)
    for name in c.GRIDS:
        row=folds[folds.outer_fold.eq(i)&folds.model.eq(name)].iloc[0]
        expected=average_precision_score(y[test],pred[name+'_score'].to_numpy()[test])
        assert abs(expected-row.pr_auc_ap)<1e-12
        scores=tuning[tuning.outer_fold.eq(i)&tuning.model.eq(name)].groupby('candidate').validation_ap.mean()
        chosen=int(scores.idxmax())
        assert json.loads(row.parameters)==c.GRIDS[name][chosen]
        assert abs(scores.max()-row.best_inner_ap)<1e-12

# Independent implementation: custom L2 and sklearn L2 should match.
train=np.setdiff1d(np.arange(len(y)),outer[0]); test=outer[0]
ours,_,_=c.fit_score('L2',{'lambda':.1},x[train],y[train],x[test],directions)
scale=StandardScaler().fit(x[train])
ref=LogisticRegression(C=1/(len(train)*.1),solver='lbfgs',tol=1e-10,max_iter=10000)
ref.fit(scale.transform(x[train]),y[train])
error=float(np.max(abs(ours-ref.predict_proba(scale.transform(x[test]))[:,1])))
assert error<1e-5,error
# Increasing a constrained feature cannot reverse its declared score direction.
probe=x[test].copy()
base,_,_=c.fit_score('Monotonic',{'lambda':.1},x[train],y[train],probe,directions)
for j,d in enumerate(directions):
    if d:
        altered=probe.copy(); altered[:,j]+=.2
        changed,_,_=c.fit_score('Monotonic',{'lambda':.1},x[train],y[train],altered,directions)
        assert (d*(changed-base)>=-1e-10).all()
for path in (c.ROOT/'Graph').glob('*.png'):
    with Image.open(path) as im: im.verify()
checks={'status':'passed','outer_assignments_exactly_match_legacy':True,'inner_test_separation_verified':True,'heldout_ap_recomputed':True,'inner_selection_recomputed':True,'tie_aware_ap_verified':True,'monotonic_perturbation_checks_passed':True,'l2_max_probability_error_vs_sklearn':error,'png_files_valid':True,'inner_training_fits':len(tuning),'outer_refits':len(folds)}
(res/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
