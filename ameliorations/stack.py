import sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
import exp
from exp import Xtr, Xte, Xtr_l, Xte_l, y, CAT, NUM, lgb, test

def cb(seed):
    return CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=5, random_seed=seed,
                              cat_features=CAT, verbose=0, thread_count=8)

ohe = lambda: ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT), ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM)])
BASE = {
    "catboost": (lambda s: cb(s), Xtr, Xte),
    "lgbm": (lambda s: lgb(s), Xtr_l, Xte_l),
    "rf": (lambda s: make_pipeline(ohe(), RandomForestClassifier(500, min_samples_leaf=3, n_jobs=8, random_state=s)), Xtr, Xte),
    "logreg": (lambda s: make_pipeline(ohe(), LogisticRegression(C=0.5, max_iter=3000)), Xtr, Xte),
}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = pd.DataFrame(index=range(len(y)), columns=list(BASE), dtype=float)
tst = pd.DataFrame(index=range(len(Xte)), columns=list(BASE), dtype=float).fillna(0.0)
for name, (mk, Xa, Xb) in BASE.items():
    for f, (tr, va) in enumerate(skf.split(Xa, y)):
        m = mk(f).fit(Xa.iloc[tr], y[tr])
        oof.loc[va, name] = m.predict_proba(Xa.iloc[va])[:, 1]
        tst[name] += m.predict_proba(Xb)[:, 1] / 5  # moyenne des 5 modèles de fold = bagging
    print(f"{name:9s} OOF acc {((oof[name] > 0.5) == y).mean():.4f}", flush=True)

print("corrélations des prédictions :\n", oof.corr().round(3))
print(f"soft voting (moyenne)      {((oof.mean(axis=1) > 0.5) == y).mean():.4f}")
meta = LogisticRegression(C=1.0)
cv_meta = StratifiedKFold(5, shuffle=True, random_state=7)
print(f"stacking (LogReg méta)     {cross_val_score(meta, oof, y, cv=cv_meta).mean():.4f}")
meta.fit(oof, y)
print("poids du méta-modèle :", dict(zip(oof.columns, meta.coef_[0].round(2))))

p = meta.predict_proba(tst)[:, 1]
out = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": p > 0.5})
out.to_csv(__file__.replace("stack.py", "sub_stack.csv"), index=False)
pv = tst[["catboost", "lgbm"]].mean(axis=1)
pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pv > 0.5}).to_csv(__file__.replace("stack.py", "sub_vote.csv"), index=False)
print(f"transportés prédits : stacking {out['Transported'].mean():.3f} | vote cb+lgbm {(pv > 0.5).mean():.3f}")
