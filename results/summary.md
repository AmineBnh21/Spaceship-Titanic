# Résumé des résultats

Score final (validation croisée 5×3) : **0.8172 ± 0.0093** — CatBoost · + TotalSpend, NoSpend

Validation par groupes : 0.8172 ± 0.0076

Temps d'exécution du notebook : 49 min (8 cœurs logiques)

## Progression du score

| Étape | Expérience retenue | Accuracy CV | Gain |
|---|---|---|---|
| Classe majoritaire | Classe majoritaire | 0.5036 ± 0.0001 | – |
| Meilleure baseline | Random Forest (colonnes brutes) | 0.7863 ± 0.0082 | +0.2826 |
| + Feature engineering | cumul : + Deck, CabinNum, Side | 0.8015 ± 0.0096 | +0.0152 |
| + Imputation | cumul : + Deck, CabinNum, Side | 0.8015 ± 0.0096 | +0.0000 |
| + Encodage | cumul : + Deck, CabinNum, Side | 0.8015 ± 0.0096 | +0.0000 |
| + Meilleur modèle | CatBoost | 0.8109 ± 0.0094 | +0.0094 |
| + Features revues avec le meilleur modèle | CatBoost · + TotalSpend, NoSpend | 0.8172 ± 0.0093 | +0.0063 |
| + Tuning | CatBoost · + TotalSpend, NoSpend | 0.8172 ± 0.0093 | +0.0000 |
| + Ensemble | moyenne de 5 seeds (CatBoost) | 0.8161 ± 0.0088 | -0.0011 |

## Comparaison des modèles

| Modèle | Accuracy CV (moyenne ± écart-type) |
|---|---|
| CatBoost | 0.8109 ± 0.0094 |
| HistGradientBoosting | 0.8079 ± 0.0094 |
| LightGBM | 0.8069 ± 0.0086 |
| XGBoost | 0.8053 ± 0.0097 |
| SVM | 0.8037 ± 0.0071 |
| Gradient Boosting | 0.8031 ± 0.0092 |
| Random Forest | 0.8020 ± 0.0099 |
| Réseau de neurones (MLP) | 0.7993 ± 0.0084 |
| Extra Trees | 0.7941 ± 0.0126 |
| Régression logistique | 0.7916 ± 0.0087 |
| KNN | 0.7910 ± 0.0080 |
| Arbre de décision | 0.7668 ± 0.0095 |
| Naive Bayes | 0.7119 ± 0.0442 |

## Effet de la normalisation

| Méthode | Régression logistique | KNN | SVM | Random Forest | LightGBM |
|---|---|---|---|---|---|
| Aucune | 0.7915 | 0.7794 | 0.7859 | 0.8015 | 0.8069 |
| StandardScaler | 0.7916 | 0.7839 | 0.8037 | 0.8017 | 0.8071 |
| MinMaxScaler | 0.7678 | 0.7523 | 0.7503 | 0.8014 | 0.8069 |
| RobustScaler | 0.7920 | 0.7910 | 0.7888 | 0.8014 | 0.8055 |
| QuantileTransformer | 0.7548 | 0.7694 | 0.7746 | 0.8018 | 0.8048 |
| PowerTransformer | 0.7569 | 0.7539 | 0.7708 | 0.8021 | 0.8070 |
| log + StandardScaler | 0.7781 | 0.7833 | 0.8028 | 0.8020 | 0.8067 |

## Effet de la PCA

| Configuration | Régression logistique | Random Forest |
|---|---|---|
| Sans PCA | 0.7916 ± 0.0087 | 0.8020 ± 0.0099 |
| PCA 90 % de variance | 0.7531 ± 0.0117 | 0.7610 ± 0.0094 |
| PCA 95 % de variance | 0.7879 ± 0.0094 | 0.7867 ± 0.0101 |
| PCA 99 % de variance | 0.7917 ± 0.0089 | 0.7886 ± 0.0093 |
| PCA sur les dépenses uniquement (3 composantes) | 0.7482 ± 0.0088 | 0.7620 ± 0.0082 |

## Soumissions

| Fichier | Étape | Accuracy CV | % prédits transportés |
|---|---|---|---|
| submissions/submission_01_baseline.csv | Random Forest (colonnes brutes) | 0.7863 ± 0.0082 | 51.6 % |
| submissions/submission_02_apres_feature_engineering.csv | cumul : + Deck, CabinNum, Side | 0.8015 ± 0.0096 | 46.6 % |
| submissions/submission_03_apres_tuning.csv | CatBoost · + TotalSpend, NoSpend | 0.8172 ± 0.0093 | 51.4 % |
| submissions/submission_04_solution_finale.csv | CatBoost · + TotalSpend, NoSpend | 0.8172 ± 0.0093 | 51.4 % |
