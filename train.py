"""Select churn models with training-only cross-validation, then evaluate holdout."""
import json
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

def pipeline(model):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))])
    preprocess = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include='number')),
        ('categorical', categorical, make_column_selector(dtype_exclude='number'))])
    return Pipeline([('preprocess', preprocess), ('model', model)])

def main():
    df = pd.read_csv('Customer-Churn-Records.csv')
    y = df['Exited']
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError('Exited must contain both classes 0 and 1 without missing values.')
    X = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Complain', 'Satisfaction Score'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    models = {
        'xgboost': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=SEED, n_jobs=1),
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=6, criterion='entropy', min_samples_leaf=5, class_weight='balanced', random_state=SEED, n_jobs=1)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in models.items():
        p = pipeline(model)
        scores = cross_validate(p, X_train, y_train, cv=cv, scoring={'ap': 'average_precision', 'roc_auc': 'roc_auc', 'recall': 'recall', 'f1': 'f1'}, error_score='raise')
        row = {'model': name}
        for metric in ['ap', 'roc_auc', 'recall', 'f1']:
            row['cv_' + metric + '_mean'] = float(scores['test_' + metric].mean())
            row['cv_' + metric + '_std'] = float(scores['test_' + metric].std())
        rows.append(row)
        fitted[name] = p.fit(X_train, y_train)
    comparison = pd.DataFrame(rows).sort_values('cv_ap_mean', ascending=False)
    winner = comparison.iloc[0]['model']
    best = fitted[winner]
    prediction = best.predict(X_test)
    probability = best.predict_proba(X_test)[:, 1]
    report = {'selected_model': winner, 'selection_metric': 'training_cv_average_precision', 'seed': SEED,
        'training_rows': len(X_train), 'test_rows': len(X_test), 'test_prevalence': float(y_test.mean()),
        'classification_report': classification_report(y_test, prediction, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, prediction).tolist(),
        'test_average_precision': average_precision_score(y_test, probability), 'test_roc_auc': roc_auc_score(y_test, probability),
        'features': X.columns.tolist()}
    out = Path('artifacts'); out.mkdir(exist_ok=True)
    comparison.to_csv(out / 'cv_comparison.csv', index=False)
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    joblib.dump(best, out / 'model.joblib')
    print(comparison.to_string(index=False)); print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
