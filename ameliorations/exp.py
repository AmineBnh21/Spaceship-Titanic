import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

DATA = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")) + "/"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
train = pd.read_csv(DATA + "train.csv"); test = pd.read_csv(DATA + "test.csv")
y = train["Transported"].astype(int).to_numpy(); n = len(train)
d = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)

d["Group"] = d["PassengerId"].str[:4].astype(int)
d["Surname"] = d["Name"].str.split().str[-1]
cab = d["Cabin"].str.split("/", expand=True)
d["Deck"], d["CabinNum"], d["Side"] = cab[0], cab[1].astype(float), cab[2]

def fill_from(col, key):
    m = d.dropna(subset=[col, key]).groupby(key)[col].agg(lambda s: s.mode().iloc[0])
    d[col] = d[col].fillna(d[key].map(m))

# --- imputation logique ---
for c in ["HomePlanet", "Deck", "Side", "Destination"]:
    fill_from(c, "Group")
fill_from("HomePlanet", "Surname")
fill_from("Deck", "Surname")
d.loc[d["HomePlanet"].isna() & d["Deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = "Europa"
d.loc[d["HomePlanet"].isna() & (d["Deck"] == "G"), "HomePlanet"] = "Earth"
tot = d[SPEND].sum(axis=1, min_count=1)
d.loc[d["CryoSleep"].isna() & (tot > 0), "CryoSleep"] = False
d.loc[d["CryoSleep"].isna() & (tot == 0) & d[SPEND].notna().all(axis=1), "CryoSleep"] = True
zero = (d["CryoSleep"] == True) | (d["Age"] <= 12)
for c in SPEND:
    d.loc[zero & d[c].isna(), c] = 0.0
d.loc[d["VIP"].isna(), "VIP"] = False
d["Age"] = d["Age"].fillna(d.groupby(["HomePlanet", "CryoSleep"])["Age"].transform("median"))
# numéro de cabine : interpolation à partir des groupes voisins sur le même pont
for dk in d["Deck"].dropna().unique():
    m = d["Deck"] == dk
    known = d[m & d["CabinNum"].notna()].sort_values("Group")
    if len(known) > 1:
        miss = m & d["CabinNum"].isna()
        d.loc[miss, "CabinNum"] = np.round(np.interp(d.loc[miss, "Group"], known["Group"], known["CabinNum"]))
for c in SPEND:
    d[c] = d[c].fillna(d.groupby("HomePlanet")[c].transform("median"))

# --- features ---
d["TotalSpend"] = d[SPEND].sum(axis=1)
d["NoSpend"] = (d["TotalSpend"] == 0).astype(int)
for c in SPEND + ["TotalSpend"]:
    d["log_" + c] = np.log1p(d[c])
d["Luxury"] = np.log1p(d["Spa"] + d["VRDeck"] + d["RoomService"])
d["Basic"] = np.log1p(d["FoodCourt"] + d["ShoppingMall"])
d["GroupSize"] = d.groupby("Group")["PassengerId"].transform("count")
d["Solo"] = (d["GroupSize"] == 1).astype(int)
d["FamilySize"] = d["Surname"].map(d["Surname"].value_counts()).fillna(1)
d["GroupNo"] = d["PassengerId"].str[5:].astype(int)
d["CabinRegion"] = (d["CabinNum"] // 300).astype("Int64").astype(str)
d["CabinShared"] = d.groupby("Cabin")["PassengerId"].transform("count").fillna(1)
d["AgeGroup"] = pd.cut(d["Age"], [-1, 12, 17, 25, 30, 50, 200], labels=False).astype(str)
d["Child"] = (d["Age"] <= 12).astype(int)
d["GroupSpend"] = np.log1p(d.groupby("Group")["TotalSpend"].transform("sum"))
d["GroupCryo"] = d.assign(c=d["CryoSleep"].astype(float)).groupby("Group")["c"].transform("mean")
d["DeckSide"] = d["Deck"].astype(str) + "_" + d["Side"].astype(str)
for c in ["CryoSleep", "VIP"]:
    d[c] = d[c].astype(float)

CAT = ["HomePlanet", "Destination", "Deck", "Side", "CabinRegion", "AgeGroup", "DeckSide"]
NUM = ["CryoSleep", "VIP", "Age", "CabinNum", "NoSpend", "Luxury", "Basic", "GroupSize", "Solo",
       "FamilySize", "GroupNo", "CabinShared", "Child", "GroupSpend", "GroupCryo"] + ["log_" + c for c in SPEND + ["TotalSpend"]]
for c in CAT:
    d[c] = d[c].fillna("NA").astype(str)
X = d[CAT + NUM]
Xtr, Xte = X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)

def cb(seed):
    return CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, random_seed=seed,
                              cat_features=CAT, verbose=0, thread_count=8)

def lgb(seed):
    return LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=24, min_child_samples=30, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.6, reg_lambda=3, random_state=seed, verbose=-1, n_jobs=8)

def lgbX(X):
    X = X.copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    return X

Xall_l = lgbX(X); Xtr_l, Xte_l = Xall_l.iloc[:n].reset_index(drop=True), Xall_l.iloc[n:].reset_index(drop=True)

if __name__ == "__main__" and sys.argv[1:] == ["cv"]:
    oof_cb, oof_lg = np.zeros(n), np.zeros(n)
    for rep in range(2):
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xtr, y):
            oof_cb[va] += cb(rep).fit(Xtr.iloc[tr], y[tr]).predict_proba(Xtr.iloc[va])[:, 1] / 2
            oof_lg[va] += lgb(rep).fit(Xtr_l.iloc[tr], y[tr]).predict_proba(Xtr_l.iloc[va])[:, 1] / 2
    acc = lambda p: ((p > 0.5) == y).mean()
    print(f"CatBoost {acc(oof_cb):.4f} | LGBM {acc(oof_lg):.4f} | blend 0.5 {acc((oof_cb + oof_lg) / 2):.4f} "
          f"| blend 0.7cb {acc(0.7 * oof_cb + 0.3 * oof_lg):.4f}")

if __name__ == "__main__" and sys.argv[1:] == ["sub"]:
    p = np.zeros(len(Xte))
    for s in range(5):
        p += 0.7 * cb(s).fit(Xtr, y).predict_proba(Xte)[:, 1] / 5
        p += 0.3 * lgb(s).fit(Xtr_l, y).predict_proba(Xte_l)[:, 1] / 5
    out = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": p > 0.5})
    out.to_csv(sys.argv[0].replace("exp.py", "sub.csv"), index=False)
    print("transportés prédits :", out["Transported"].mean())
