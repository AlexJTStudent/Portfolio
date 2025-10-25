import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    return (
        RandomForestRegressor,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    wo1_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv")
    wo2_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w02.csv")
    wo3_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w03.csv")
    wo4_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w04.csv")
    wo5_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w05.csv")
    wo6_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w06.csv")
    wo7_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w07.csv")
    wo8_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w08.csv")
    wo9_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w09.csv")
    wo10_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w10.csv")
    wo11_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w11.csv")
    wo12_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w12.csv")
    wo13_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w13.csv")
    wo14_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w14.csv")
    wo15_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w15.csv")
    wo16_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w16.csv")
    wo17_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w17.csv")
    wo18_pre = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w18.csv")
    return (wo1_pre,)


@app.cell
def _(wo1_pre):
    wo1_pre.head(3)
    return


@app.cell
def _(pd):
    wo1_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w01.csv")
    wo2_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w02.csv")
    wo3_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w03.csv")
    wo4_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w04.csv")
    wo5_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w05.csv")
    wo6_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w06.csv")
    wo7_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w07.csv")
    wo8_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w08.csv")
    wo9_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w09.csv")
    wo10_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w10.csv")
    wo11_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w11.csv")
    wo12_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w12.csv")
    wo13_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w13.csv")
    wo14_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w14.csv")
    wo15_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w15.csv")
    wo16_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w16.csv")
    wo17_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w17.csv")
    wo18_pos = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w18.csv")
    return (wo1_pos,)


@app.cell
def _(wo1_pos):
    wo1_pos.head(3)
    return


@app.cell
def _(test):
    test.head(3)
    return


@app.cell
def _(pd):
    test = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/test.csv")
    return (test,)


@app.cell
def _(
    RandomForestRegressor,
    mean_absolute_error,
    train_test_split,
    wo1_pos,
    wo1_pre,
):

    data = wo1_pre.merge(
        wo1_pos,
        on=["game_id", "play_id", "nfl_id", "frame_id"],
        suffixes=("_pre", "_pos"))


    feature_cols = [
        "x_pre", "y_pre", "s", "a", "dir", "o",
        "absolute_yardline_number", "ball_land_x", "ball_land_y"
    ]
    X = data[feature_cols]
    y = data[["x_pos", "y_pos"]]


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    model_x = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_y = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_x.fit(X_train, y_train["x_pos"])
    model_y.fit(X_train, y_train["y_pos"])


    x_pred_val = model_x.predict(X_val)
    y_pred_val = model_y.predict(X_val)
    mae_x = mean_absolute_error(y_val["x_pos"], x_pred_val)
    mae_y = mean_absolute_error(y_val["y_pos"], y_pred_val)
    print(f"Validation MAE → x: {mae_x:.3f}, y: {mae_y:.3f}")
    return feature_cols, model_x, model_y, x_pred_val, y_pred_val, y_val


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    np,
    pd,
    r2_score,
    x_pred_val,
    y_pred_val,
    y_val,
):
    # --- 6. Metrics ---
    metrics = {
        "MAE": [
            mean_absolute_error(y_val["x_pos"], x_pred_val),
            mean_absolute_error(y_val["y_pos"], y_pred_val)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_val["x_pos"], x_pred_val)),
            np.sqrt(mean_squared_error(y_val["y_pos"], y_pred_val))
        ],
        "R2": [
            r2_score(y_val["x_pos"], x_pred_val),
            r2_score(y_val["y_pos"], y_pred_val)
        ]
    }
    metrics_table = pd.DataFrame(metrics, index=["x_position", "y_position"])
    print(metrics_table)

    return


@app.cell
def _(feature_cols, model_x, model_y, test, wo1_pre):
    # --- 6. Prepare test set ---
    # Rename x,y in wo1_pre to avoid collisions
    wo1_pre_renamed = wo1_pre.rename(columns={"x": "x_pre", "y": "y_pre"})

    test_data = test.merge(
        wo1_pre_renamed,
        on=["game_id", "play_id", "nfl_id", "frame_id"],
        how="left"
    )

    # --- 7. Prepare features for prediction ---
    X_test_final = test_data[feature_cols]

    # --- 8. Predict ---
    test_data["x_pred"] = model_x.predict(X_test_final)
    test_data["y_pred"] = model_y.predict(X_test_final)

    # --- 9. Export predictions ---
    submission = test_data[[
        "game_id", "play_id", "nfl_id", "frame_id", "x_pred", "y_pred"
    ]]
    submission.to_csv("submission.csv", index=False)

    print("✅ Predictions saved to submission.csv")
    return


@app.cell
def _(plt, x_pred_val, y_val):
    # Scatter plot for X positions
    plt.figure(figsize=(10, 5))
    plt.scatter(y_val["x_pos"], x_pred_val, alpha=0.5, label="x")
    plt.plot([y_val["x_pos"].min(), y_val["x_pos"].max()],
             [y_val["x_pos"].min(), y_val["x_pos"].max()],
             color='red', linestyle='--', label="Perfect Prediction")
    plt.xlabel("Actual X")
    plt.ylabel("Predicted X")
    plt.title("Predicted vs Actual X Positions")
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, y_pred_val, y_val):
    # Scatter plot for Y positions
    plt.figure(figsize=(10, 5))
    plt.scatter(y_val["y_pos"], y_pred_val, alpha=0.5, label="y")
    plt.plot([y_val["y_pos"].min(), y_val["y_pos"].max()],
             [y_val["y_pos"].min(), y_val["y_pos"].max()],
             color='red', linestyle='--', label="Perfect Prediction")
    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title("Predicted vs Actual Y Positions")
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, x_pred_val, y_pred_val, y_val):
    # Optional: combined X vs Y comparison
    plt.figure(figsize=(7, 7))
    plt.scatter(y_val["x_pos"], y_val["y_pos"], alpha=0.3, label="Actual", color='blue')
    plt.scatter(x_pred_val, y_pred_val, alpha=0.3, label="Predicted", color='orange')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Actual vs Predicted Player Positions")
    plt.legend()
    plt.show()
    return


@app.cell
def _(feature_cols, model_x, np, plt):
    # --- Feature importance for X model ---
    importances_x = model_x.feature_importances_
    indices_x = np.argsort(importances_x)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances for X Position")
    plt.bar(range(len(feature_cols)), importances_x[indices_x], align="center")
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices_x], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()




    return (importances_x,)


@app.cell
def _(feature_cols, model_y, np, plt):
    # --- Feature importance for Y model ---
    importances_y = model_y.feature_importances_
    indices_y = np.argsort(importances_y)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances for Y Position")
    plt.bar(range(len(feature_cols)), importances_y[indices_y], align="center")
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices_y], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
    return (importances_y,)


@app.cell
def _(feature_cols, importances_x, importances_y, pd):
    # --- Optional: combined importance table ---
    importance_table = pd.DataFrame({
        "Feature": feature_cols,
        "Importance_X": importances_x,
        "Importance_Y": importances_y
    }).sort_values(by="Importance_X", ascending=False)

    print("\n=== Feature Importances ===")
    print(importance_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""make a case study for a single play showing how the predicted compares to actual play""")
    return


if __name__ == "__main__":
    app.run()
