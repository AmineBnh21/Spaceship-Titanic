# Spaceship Titanic — Comparaison systématique des techniques de Machine Learning

Prédiction des passagers « transportés » dans une autre dimension (compétition Kaggle), en testant **une par une** les techniques classiques du machine learning (feature engineering, imputation, encodage, normalisation, PCA, sélection de features, 13 modèles, tuning, ensembles) pour mesurer ce que chacune apporte réellement.

**Stack :** Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna

**Score final :** 0,817 ± 0,009 en validation croisée (CatBoost), contre 0,786 pour la baseline · **0,803** au classement public Kaggle

## Installation

```bash
pip install -r requirements.txt
```

Les données ne sont pas versionnées. Télécharger le dataset sur la [page de la compétition](https://www.kaggle.com/competitions/spaceship-titanic/data) et placer les fichiers dans `data/` :

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Puis exécuter `spaceship_titanic.ipynb` de haut en bas. L'exécution complète prend **≈ 50 min sur CPU** (8 cœurs), dont ≈ 14 min de tuning Optuna. Elle produit :

- `results/experiments.csv` : le journal des 114 expériences ;
- `results/summary.md` : les tableaux de résultats ;
- `results/figures/` : les graphiques ;
- `submissions/` : 4 fichiers de soumission (baseline, après feature engineering, après tuning, solution finale).

## 1. Le problème

En l'an 2912, le vaisseau *Spaceship Titanic* percute une anomalie spatio-temporelle, et environ la moitié des passagers sont transportés dans une autre dimension.

**Objectif :** à partir des informations sur chaque passager (planète d'origine, cabine, âge, dépenses à bord…), prédire s'il a été transporté.

C'est une **classification binaire**, évaluée à l'**accuracy**. Les classes sont équilibrées (50,4 % de transportés), donc l'accuracy est une métrique fiable, et le plancher (prédire toujours la même classe) est d'environ 50 %.

## 2. Le dataset

- **8 693** passagers dans le train (avec la cible), **4 277** dans le test
- 13 colonnes, environ **2 % de valeurs manquantes** par colonne (24 % des lignes ont au moins un trou)

| Colonne | Signification | Ce qu'on peut en tirer |
|---|---|---|
| `PassengerId` | Identifiant `gggg_pp` | `gggg` = numéro de groupe (famille, amis) |
| `HomePlanet` | Planète d'origine | Earth, Europa, Mars |
| `CryoSleep` | Sommeil cryogénique | Endormi dans sa cabine tout le voyage |
| `Cabin` | Cabine `pont/numéro/côté` | Position dans le vaisseau |
| `Destination` | Planète de destination | 3 destinations |
| `Age`, `VIP` | Âge, service VIP | – |
| `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` | Dépenses à bord | Activité du passager |
| `Name` | Prénom et nom | Nom de famille → familles |
| `Transported` | **La cible** | – |

**Ce que montre l'exploration :**

- **Les dépenses sont très liées à la cible.** Parmi ceux qui n'ont rien dépensé dans un service, 59 à 64 % sont transportés, contre 26 à 35 % parmi ceux qui ont dépensé.
- **Des règles déterministes** : un passager en cryosommeil ne dépense jamais rien, un enfant de 12 ans ou moins non plus. Les enfants sont transportés à 70 % (contre 48 % pour les autres).
- **Les groupes sont cohérents** : tous les membres d'un groupe viennent de la même planète, et aucun groupe n'est à la fois dans le train et dans le test.

## 3. Démarche

Le principe du projet : **ne rien supposer, tout mesurer.** Beaucoup de tutoriels appliquent la normalisation, la PCA ou un modèle précis « parce que c'est ce qu'on fait ». Ici, chaque technique est testée seule, et on ne garde que ce qui améliore réellement le score.

```mermaid
flowchart LR
    A[0. Exploration] --> B[1. Baselines] --> C[2. Feature engineering] --> D[3. Imputation] --> E[4. Encodage]
    E --> F[5. Normalisation] --> G[6. PCA / sélection] --> H[7. 13 modèles] --> I[8. Tuning Optuna] --> J[9. Ensembles] --> K[10. Solution finale]
```

**Protocole :**

- **Validation croisée stratifiée 5 folds × 3 répétitions**, avec les **mêmes folds** pour toutes les expériences. Le score est la moyenne ± écart-type des 15 folds.
- **Un seul changement par expérience**, par rapport à la meilleure configuration du moment : on sait exactement d'où vient chaque gain.
- **Aucune fuite de données** : imputation, encodage, normalisation, PCA et sélection de features sont calculés **dans chaque fold**, sur la partie entraînement uniquement (`Pipeline` scikit-learn).
- **Règle de décision** : une technique n'est retenue que si son gain dépasse le bruit (comparaison fold par fold : nombre de folds gagnés sur 15). Un « progrès » de 0,1 point peut n'être que du bruit.
- **Décisions sur la validation croisée uniquement**, jamais sur le classement Kaggle.

**Les techniques testées :**

| Phase | Ce qui est comparé |
|---|---|
| 2. Feature engineering | `Deck/CabinNum/Side` (cabine), `GroupSize/IsAlone`, `TotalSpend/NoSpend`, `AgeGroup/IsChild`, `FamilySize`, indicateurs de valeurs manquantes — chacun seul, puis en cumul |
| 3. Imputation | médiane, moyenne, constante, `KNNImputer`, `IterativeImputer`, aucune (NaN natifs des boostings), imputation logique (cryosommeil → dépenses = 0, etc.) |
| 4. Encodage | one-hot, ordinal, target encoding, fréquence, catégories natives LightGBM / CatBoost |
| 5. Normalisation | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `QuantileTransformer`, `PowerTransformer`, log + standard — sur 3 modèles sensibles à l'échelle et 2 modèles d'arbres |
| 6. PCA et sélection | PCA à 90 / 95 / 99 % de variance, PCA sur les dépenses seules ; sélection par importance, `RFECV`, permutation importance |
| 7. Modèles | Régression logistique, KNN, SVM, Naive Bayes, arbre de décision, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, MLP |
| 8. Tuning | Optuna (TPE + pruning), 30 essais sur chacun des 3 meilleurs modèles |
| 9. Ensembles | moyenne, moyenne + SVM, moyenne pondérée (poids choisis hors fold), stacking, moyenne de 5 seeds |
| 10. Vérifications | validation par groupes (`StratifiedGroupKFold`), ablation du feature engineering |

## 4. Résultats

Accuracy en validation croisée 5×3 (moyenne ± écart-type sur 15 folds). 114 expériences au total, toutes dans `results/experiments.csv`.

### Progression du score

| Étape | Configuration retenue | Accuracy CV | Gain |
|---|---|---|---|
| Classe majoritaire | – | 0,504 ± 0,000 | – |
| Baseline | Random Forest, colonnes brutes | 0,786 ± 0,008 | +28,3 pts |
| + Feature engineering | + `Deck`, `CabinNum`, `Side` | 0,802 ± 0,010 | **+1,5 pt** |
| + Imputation | inchangée (médiane) | 0,802 ± 0,010 | 0 |
| + Encodage | inchangé (one-hot) | 0,802 ± 0,010 | 0 |
| + Meilleur modèle | CatBoost | 0,811 ± 0,009 | **+0,9 pt** |
| + Features revues avec CatBoost | + `TotalSpend`, `NoSpend` | **0,817 ± 0,009** | **+0,6 pt** |
| + Tuning | non retenu | 0,817 ± 0,009 | 0 |
| + Ensemble | non retenu | 0,817 ± 0,009 | 0 |

### Comparaison des 13 modèles

Mêmes features pour tous, chacun avec le prétraitement qui lui convient (phases 3 à 6).

| Modèle | Accuracy CV |
|---|---|
| **CatBoost** | **0,811 ± 0,009** |
| HistGradientBoosting | 0,808 ± 0,009 |
| LightGBM | 0,807 ± 0,009 |
| XGBoost | 0,805 ± 0,010 |
| SVM | 0,804 ± 0,007 |
| Gradient Boosting | 0,803 ± 0,009 |
| Random Forest | 0,802 ± 0,010 |
| Réseau de neurones (MLP) | 0,799 ± 0,008 |
| Extra Trees | 0,794 ± 0,013 |
| Régression logistique | 0,792 ± 0,009 |
| KNN | 0,791 ± 0,008 |
| Arbre de décision | 0,767 ± 0,009 |
| Naive Bayes | 0,712 ± 0,044 |

### Effet de la normalisation

| Méthode | Rég. logistique | KNN | SVM | Random Forest | LightGBM |
|---|---|---|---|---|---|
| Aucune | 0,792 | 0,779 | 0,786 | 0,802 | 0,807 |
| StandardScaler | 0,792 | 0,784 | **0,804** | 0,802 | 0,807 |
| MinMaxScaler | 0,768 | 0,752 | 0,750 | 0,801 | 0,807 |
| RobustScaler | 0,792 | **0,791** | 0,789 | 0,801 | 0,806 |
| QuantileTransformer | 0,755 | 0,769 | 0,775 | 0,802 | 0,805 |
| PowerTransformer | 0,757 | 0,754 | 0,771 | 0,802 | 0,807 |
| log + StandardScaler | 0,778 | 0,783 | 0,803 | 0,802 | 0,807 |

### Effet de la PCA

| Configuration | Rég. logistique | Random Forest |
|---|---|---|
| Sans PCA | **0,792 ± 0,009** | **0,802 ± 0,010** |
| PCA 90 % de variance | 0,753 ± 0,012 | 0,761 ± 0,009 |
| PCA 95 % de variance | 0,788 ± 0,009 | 0,787 ± 0,010 |
| PCA 99 % de variance | 0,792 ± 0,009 | 0,789 ± 0,009 |
| PCA sur les dépenses (3 composantes) | 0,748 ± 0,009 | 0,762 ± 0,008 |

### Tuning et ensembles (référence : CatBoost par défaut, 0,817)

| Configuration | Accuracy CV | Verdict |
|---|---|---|
| CatBoost tuné (Optuna, 30 essais) | 0,817 ± 0,008 | dans le bruit (7/15 folds) |
| HistGradientBoosting tuné | 0,812 ± 0,009 | dans le bruit |
| LightGBM tuné | 0,811 ± 0,011 | dans le bruit |
| Moyenne des 3 meilleurs modèles | 0,815 ± 0,009 | perte (2/15 folds) |
| Moyenne pondérée (poids hors fold) | 0,815 ± 0,009 | perte (5/15 folds) |
| Stacking (régression logistique) | 0,816 ± 0,009 | dans le bruit |
| Moyenne de 5 seeds CatBoost | 0,816 ± 0,009 | dans le bruit |

### Vérifications finales

| Test | Accuracy CV |
|---|---|
| Solution finale (CatBoost + features), validation classique | 0,817 ± 0,009 |
| Même modèle, validation par groupes (`StratifiedGroupKFold`) | 0,817 ± 0,008 |
| Ablation : CatBoost tuné, colonnes brutes seulement | 0,794 ± 0,010 (−2,3 pts) |

### Soumissions Kaggle

| Fichier | Configuration | Accuracy CV | Score public Kaggle |
|---|---|---|---|
| `submission_01_baseline.csv` | Random Forest, colonnes brutes | 0,786 | *à compléter* |
| `submission_02_apres_feature_engineering.csv` | RF + features cabine | 0,802 | *à compléter* |
| `submission_04_solution_finale.csv` | CatBoost + features | 0,817 | **0,803** |

(`submission_03_apres_tuning.csv` est identique à la solution finale, le tuning n'ayant pas été retenu.)

### Analyse

- **Le feature engineering est le levier principal.** Il apporte +2,3 pts au modèle final (ablation), contre +0,9 pt pour le choix du modèle et 0 pour le tuning. La seule feature qui s'impose sur le modèle de référence est la **cabine** (pont / numéro / côté) : +1,5 pt, gagnante sur les **15 folds sans exception**.
- **Des features « évidentes » ne servent à rien, ou nuisent.** Ajoutées seules au Random Forest, `GroupSize` (−1,0 pt) et `FamilySize` (−1,1 pt) dégradent le score. `TotalSpend` / `NoSpend` n'aident pas le Random Forest, mais apportent +0,6 pt à CatBoost (14/15 folds) : l'utilité d'une feature dépend du modèle, d'où la revue des features avec le meilleur modèle en phase 7.
- **L'imputation et l'encodage ne changent presque rien.** Avec 2 % de valeurs manquantes et 2 à 8 modalités par variable catégorielle, toutes les méthodes sont à ±0,3 pt de la médiane / du one-hot, dans le bruit. L'imputation « logique », séduisante sur le papier, ne passe pas la règle de décision. Les catégories natives de CatBoost gagnent +0,1 pt mais multiplient le temps d'entraînement par 6,5.
- **La normalisation n'aide que les modèles sensibles à l'échelle.** Pour les arbres, aucune méthode ne bouge le score de plus de 0,2 pt. Pour le SVM, `StandardScaler` apporte +1,8 pt (15/15 folds). Le mauvais choix coûte cher : `MinMaxScaler` fait perdre jusqu'à 5,3 pts au SVM, car les dépenses ont des valeurs extrêmes (jusqu'à 30 000) qui écrasent toutes les autres valeurs près de 0.
- **La PCA n'aide jamais.** Pour le Random Forest, toutes les variantes perdent de 1,3 à 4,1 pts, sur les 15 folds. La PCA garde les directions de plus grande variance, qui ne sont pas celles qui prédisent la cible ; elle détruit en plus les seuils simples (« a dépensé ou non ») que les arbres exploitent.
- **Le boosting gagne, mais de peu.** Les 4 boostings sont en tête (0,805–0,811), mais le SVM, le Gradient Boosting et le Random Forest sont à moins de 1 pt, dans l'écart-type. Naive Bayes s'effondre (0,712) : ses hypothèses d'indépendance sont fausses ici (dépenses et cryosommeil sont liés).
- **Ni le tuning ni les ensembles n'améliorent le score.** Optuna trouve des configurations légèrement meilleures sur ses propres essais, mais le gain ne se confirme pas sur les 15 folds : on garde les hyperparamètres par défaut. Les ensembles échouent car les modèles se ressemblent trop : les 3 boostings ne sont en désaccord que sur 4,6 à 5,8 % des passagers. Le SVM est plus différent (≈ 10 % de désaccord) mais trop faible pour aider la moyenne.
- **Le score n'est pas gonflé par les groupes.** Garder chaque groupe entier dans un même fold donne exactement le même score (0,817), avec un écart-type plus faible : le modèle n'exploite pas une fuite d'information entre membres d'un même groupe.
- **Leçon.** Sur un petit dataset tabulaire, la compréhension des données (ici, la structure de la cabine) rapporte plus que le choix ou le réglage du modèle ; et un protocole strict évite de retenir des « améliorations » qui ne sont que du bruit.

## 5. Limites

- **Petit dataset.** Avec 8 693 passagers, l'écart-type entre folds (≈ 0,9 pt) est du même ordre que les différences entre les meilleures techniques : beaucoup de conclusions sont « dans le bruit ».
- **Écart entre validation croisée et classement public.** La solution finale fait 0,817 en validation croisée mais 0,803 au classement public (−1,4 pt). Le classement public n'est calculé que sur une partie du test, et le choix de la configuration parmi 114 expériences sur la validation croisée rend celle-ci légèrement optimiste.
- **Tuning limité.** 30 essais Optuna par modèle ; un budget plus large pourrait trouver un gain réel, mais probablement faible au vu du plateau observé.
- **Ordre des phases.** Les décisions des phases 2 à 6 ont été prises avec un Random Forest ; elles ont été revues avec CatBoost en phase 7, mais pas toutes les combinaisons possibles.
- **Données fictives.** Le dataset a été généré pour la compétition ; certaines relations (cryosommeil → aucune dépense) sont plus nettes que dans des données réelles.

## 6. Pistes d'amélioration

- Exploiter davantage la structure des groupes : propager les informations connues d'un membre aux autres (planète, cryosommeil, cabine).
- Features issues des noms de famille en croisant train et test.
- Plus de budget de tuning, ou un tuning conjoint des features et des hyperparamètres.
- Ensembles de modèles plus différents (par exemple, un réseau de neurones tabulaire adapté).

## 7. Structure du projet

```
.
├── README.md
├── spaceship_titanic.ipynb   # tout le projet, avec les résultats affichés
├── requirements.txt
├── data/                     # données Kaggle (non versionnées)
├── submissions/              # fichiers de soumission
└── results/
    ├── experiments.csv       # journal des 114 expériences
    ├── summary.md            # tableaux de résultats
    └── figures/              # graphiques
```

## 8. Références

- Kaggle, *Spaceship Titanic* — https://www.kaggle.com/competitions/spaceship-titanic
- Pedregosa F. et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.
- Chen T. & Guestrin C., *XGBoost: A Scalable Tree Boosting System*, KDD 2016.
- Ke G. et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017.
- Prokhorenkova L. et al., *CatBoost: unbiased boosting with categorical features*, NeurIPS 2018.
- Akiba T. et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
