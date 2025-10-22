"""
---
title: "Insurance Claims Prediction Model"
execute:
  echo: false
format:
  html:
    theme: flatly
    code-fold: true  
    code-tools: true
---
"""


import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    return (
        ColumnTransformer,
        OneHotEncoder,
        Pipeline,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        classification_report,
        np,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    RandomForestClassifier,
    StandardScaler,
    accuracy_score,
    classification_report,
    pd,
    train_test_split,
):
    # 1. Load and preprocess
    data = pd.read_csv("Projects/train.csv").drop(columns=["policy_id"])
    data['max_torque_value'] = data['max_torque'].str.extract(r'([\d\.]+)Nm').astype(float)
    data['max_power_value'] = data['max_power'].str.extract(r'([\d\.]+)bhp').astype(float)
    data.drop(columns=['max_torque','max_power'], inplace=True)

    # Convert Yes/No to 0/1
    binary_cols = [c for c in data.columns if data[c].isin(['Yes','No']).all()]
    data[binary_cols] = data[binary_cols].apply(lambda x: x.map({'Yes':1,'No':0}))

    # 2. Features and target
    X = data.drop(columns=['is_claim'])
    y = data['is_claim']

    # 3. Column types
    categorical_cols = X.select_dtypes('object').columns.tolist()
    numeric_cols = X.select_dtypes(['int64','float64']).columns.tolist()

    # 4. Pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
        ))
    ])

    # 5. Split, train, predict
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_eval)

    # 6. Metrics table
    metrics_df = pd.DataFrame(classification_report(y_eval, y_pred, output_dict=True)).transpose()
    metrics_df.loc['accuracy', ['precision','recall','f1-score','support']] = [accuracy_score(y_eval, y_pred), None, None, len(y_eval)]
    metrics_df = metrics_df[['precision','recall','f1-score','support']].round(2)

    print(metrics_df)
    return categorical_cols, numeric_cols, rf_pipeline


@app.cell
def _(categorical_cols, np, numeric_cols, pd, plt, rf_pipeline):

    # Extract trained Random Forest and feature names
    rf_model = rf_pipeline.named_steps['classifier']

    # Numeric columns
    num_features = numeric_cols

    # One-hot encoded categorical columns excluding area_cluster
    cat_features = [
        f for f in rf_pipeline.named_steps['preprocessor']
            .named_transformers_['cat'].get_feature_names_out(categorical_cols)
        if not f.startswith('area_cluster_')
    ]

    # 1. Combine all feature names (including area_cluster)
    all_features = np.concatenate([
        numeric_cols,
        rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    ])
    importances = rf_model.feature_importances_

    # 2. Create DataFrame
    feat_df = pd.DataFrame({'feature': all_features, 'importance': importances})

    # 3. Map one-hot columns to original feature
    def map_original(col):
        for c in categorical_cols:
            if col.startswith(c + '_'):
                return c
        return col

    feat_df['original_feature'] = feat_df['feature'].apply(map_original)

    # 4. Exclude area_cluster
    feat_df = feat_df[feat_df['original_feature'] != 'area_cluster']

    # 5. Aggregate importances by original feature
    agg_feat_df = feat_df.groupby('original_feature')['importance'].sum().sort_values(ascending=False).reset_index()

    # 6. Plot top 20
    top20 = agg_feat_df.head(20)
    plt.figure(figsize=(10,6))
    plt.barh(top20['original_feature'][::-1], top20['importance'][::-1], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances (Excluding area_cluster)')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
