import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    return (
        DataLoader,
        StandardScaler,
        TensorDataset,
        mean_absolute_error,
        mean_squared_error,
        nn,
        np,
        optim,
        pd,
        plt,
        r2_score,
        torch,
        train_test_split,
    )


@app.cell
def _(np, torch):
    torch.manual_seed(42)
    np.random.seed(42)
    return


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(pd):
    print("Loading all 18 weeks of data...")
    input_files = [pd.read_csv(f"Projects/NFL/nfl-big-data-bowl-2026-prediction/train/input_2023_w{week:02d}.csv") for week in range(1, 19)]
    output_files = [pd.read_csv(f"Projects/NFL/nfl-big-data-bowl-2026-prediction/train/output_2023_w{week:02d}.csv") for week in range(1, 19)]
    all_input = pd.concat(input_files, ignore_index=True)
    all_output = pd.concat(output_files, ignore_index=True)
    print(f"Total input rows: {len(all_input):,}")
    print(f"Total output rows: {len(all_output):,}")
    return all_input, all_output


@app.cell
def _(
    DataLoader,
    StandardScaler,
    TensorDataset,
    all_input,
    all_output,
    device,
    nn,
    pd,
    torch,
    train_test_split,
):
    # Merge and prepare data
    print("Merging data...")
    data = all_input.merge(all_output, on=["game_id", "play_id", "nfl_id", "frame_id"], suffixes=("_pre", "_pos"))

    # One-hot encode categorical features
    position_dummies = pd.get_dummies(data['player_position'], prefix='pos')
    side_dummies = pd.get_dummies(data['player_side'], prefix='side')
    role_dummies = pd.get_dummies(data['player_role'], prefix='role')

    print(f"\nCreated dummy variables:")
    print(f"  - Position dummies ({len(position_dummies.columns)}): {position_dummies.columns.tolist()}")
    print(f"  - Side dummies ({len(side_dummies.columns)}): {side_dummies.columns.tolist()}")
    print(f"  - Role dummies ({len(role_dummies.columns)}): {role_dummies.columns.tolist()}")

    # Numerical features
    numerical_features = ["x_pre", "y_pre", "s", "a", "dir", "o", "absolute_yardline_number", "ball_land_x", "ball_land_y"]
    X_combined = pd.concat([data[numerical_features], position_dummies, side_dummies, role_dummies], axis=1)
    feature_cols = X_combined.columns.tolist()

    print(f"\nTotal features: {len(feature_cols)}")

    X = X_combined.values
    y = data[["x_pos", "y_pos"]].values

    # Train/val split & scaling
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to PyTorch
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # Define model
    class PositionPredictor(nn.Module):
        def __init__(self, input_dim):
            super(PositionPredictor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 2)
            )
        def forward(self, x):
            return self.network(x)
    return (
        PositionPredictor,
        X_val_tensor,
        data,
        feature_cols,
        numerical_features,
        position_dummies,
        role_dummies,
        scaler,
        side_dummies,
        train_loader,
        y_val,
        y_val_tensor,
    )


@app.cell
def _(PositionPredictor, device, feature_cols, nn, optim):
    input_dim = len(feature_cols)
    model = PositionPredictor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return criterion, model, optimizer, scheduler


@app.cell
def _(
    X_val_tensor,
    criterion,
    model,
    optimizer,
    scheduler,
    torch,
    train_loader,
    y_val_tensor,
):
    # Training loop
    print("\nTraining model...")
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses


@app.cell
def _(
    X_val_tensor,
    mean_absolute_error,
    mean_squared_error,
    model,
    np,
    pd,
    r2_score,
    torch,
    y_val,
):
    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor).cpu().numpy()

    x_pred_val = predictions[:, 0]
    y_pred_val = predictions[:, 1]

    metrics = {
        "MAE": [
            mean_absolute_error(y_val[:, 0], x_pred_val),
            mean_absolute_error(y_val[:, 1], y_pred_val)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_val[:, 0], x_pred_val)),
            np.sqrt(mean_squared_error(y_val[:, 1], y_pred_val))
        ],
        "R2": [
            r2_score(y_val[:, 0], x_pred_val),
            r2_score(y_val[:, 1], y_pred_val)
        ]
    }
    metrics_df = pd.DataFrame(metrics, index=["x_position", "y_position"])
    print("\n=== Model Performance Metrics ===")
    print(metrics_df)
    return x_pred_val, y_pred_val


@app.cell
def _(
    X_val_tensor,
    criterion,
    data,
    device,
    feature_cols,
    mean_absolute_error,
    model,
    np,
    numerical_features,
    pd,
    plt,
    position_dummies,
    role_dummies,
    scaler,
    side_dummies,
    torch,
    train_losses,
    val_losses,
    x_pred_val,
    y_pred_val,
    y_val,
    y_val_tensor,
):
    # Visualizations
    print("\nGenerating visualizations...")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(y_val[:, 0], x_pred_val, alpha=0.3, s=10)
    plt.plot([y_val[:, 0].min(), y_val[:, 0].max()], [y_val[:, 0].min(), y_val[:, 0].max()], 
             color='red', linestyle='--', linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual X Position")
    plt.ylabel("Predicted X Position")
    plt.title("Predicted vs Actual X Positions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('x_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(y_val[:, 1], y_pred_val, alpha=0.3, s=10)
    plt.plot([y_val[:, 1].min(), y_val[:, 1].max()], [y_val[:, 1].min(), y_val[:, 1].max()], 
             color='red', linestyle='--', linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual Y Position")
    plt.ylabel("Predicted Y Position")
    plt.title("Predicted vs Actual Y Positions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('y_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    ###
    # 2D position comparison with football field
    plt.figure(figsize=(12, 6.4))
    field_img = plt.imread('Projects/NFL/nfl-big-data-bowl-2026-prediction/field.png')
    plt.imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0)
    sample_indices = np.random.choice(len(y_val), size=min(1000, len(y_val)), replace=False)
    plt.scatter(y_val[sample_indices, 0], y_val[sample_indices, 1], 
                alpha=0.6, s=50, label="Actual", color='blue', edgecolors='white', linewidths=0.5, zorder=2)
    plt.scatter(x_pred_val[sample_indices], y_pred_val[sample_indices], 
                alpha=0.6, s=50, label="Predicted", color='orange', edgecolors='white', linewidths=0.5, zorder=2)
    plt.xlabel("X Position (yards)", fontsize=12)
    plt.ylabel("Y Position (yards)", fontsize=12)
    plt.title("Actual vs Predicted Player Positions on Field (Sample)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=14, markerscale=2, framealpha=0.9)
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.savefig('position_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    # Test predictions
    print("\nPreparing test predictions...")
    test_meta = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/test.csv")
    test_input = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/test_input.csv")
    print(f"test.csv shape: {test_meta.shape}")
    print(f"test_input.csv shape: {test_input.shape}")

    test_data = test_meta.merge(test_input, on=["game_id", "play_id", "nfl_id", "frame_id"], how="left")
    if 'x' in test_data.columns and 'x_pre' not in test_data.columns:
        test_data = test_data.rename(columns={"x": "x_pre", "y": "y_pre"})

    print(f"Merged test data shape: {test_data.shape}")
    print("\n=== Missing Values in Test Data ===")
    missing_count = 0
    for col in numerical_features:
        if col in test_data.columns:
            test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
            null_count = test_data[col].isnull().sum()
            if null_count > 0:
                train_mean = data[col].mean()
                test_data[col] = test_data[col].fillna(train_mean)
                print(f"  {col}: {null_count} missing → filled with mean {train_mean:.4f}")
                missing_count += null_count
    if missing_count == 0:
        print("  ✅ No missing values!")

    # Create dummy variables
    test_position_dummies = pd.get_dummies(test_data['player_position'], prefix='pos')
    test_side_dummies = pd.get_dummies(test_data['player_side'], prefix='side')
    test_role_dummies = pd.get_dummies(test_data['player_role'], prefix='role')

    for col in position_dummies.columns:
        if col not in test_position_dummies.columns:
            test_position_dummies[col] = 0
    for col in side_dummies.columns:
        if col not in test_side_dummies.columns:
            test_side_dummies[col] = 0
    for col in role_dummies.columns:
        if col not in test_role_dummies.columns:
            test_role_dummies[col] = 0

    test_position_dummies = test_position_dummies[position_dummies.columns]
    test_side_dummies = test_side_dummies[side_dummies.columns]
    test_role_dummies = test_role_dummies[role_dummies.columns]

    test_numerical = test_data[numerical_features].astype(np.float64)
    test_combined = pd.concat([
        test_numerical.reset_index(drop=True),
        test_position_dummies.reset_index(drop=True),
        test_side_dummies.reset_index(drop=True),
        test_role_dummies.reset_index(drop=True)
    ], axis=1)

    X_test = test_combined[feature_cols].values.astype(np.float64)
    print(f"\nTest features shape: {X_test.shape}")
    print(f"Expected shape: ({len(test_data)}, {len(feature_cols)})")

    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy()

    if np.isnan(test_predictions).any():
        print(f"⚠️  WARNING: NaN values in predictions!")
    else:
        print("✅ All predictions are valid (no NaN)")

    x_std = np.std(test_predictions[:, 0])
    y_std = np.std(test_predictions[:, 1])
    print(f"\n=== Prediction Quality Check ===")
    print(f"X predictions - Mean: {test_predictions[:, 0].mean():.4f}, Std: {x_std:.4f}")
    print(f"Y predictions - Mean: {test_predictions[:, 1].mean():.4f}, Std: {y_std:.4f}")

    test_data["x_pred"] = test_predictions[:, 0]
    test_data["y_pred"] = test_predictions[:, 1]
    submission = test_data[["id", "x_pred", "y_pred"]]
    submission.to_csv("submission.csv", index=False)

    print(f"\n✅ Predictions saved to submission.csv")
    print(f"✅ Submission shape: {submission.shape}")
    print(f"\nFirst 10 predictions:")
    print(submission.head(10))
    ###
    # Feature importance analysis
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    print(f"\nCalculating permutation importance on validation set...")
    print(f"Validation samples: {len(X_val_tensor)}")

    model.eval()
    with torch.no_grad():
        baseline_pred = model(X_val_tensor)
        baseline_loss = criterion(baseline_pred, y_val_tensor).item()

    print(f"Baseline validation loss: {baseline_loss:.4f}")

    importances = []
    print(f"Processing {len(feature_cols)} features...")
    for fi, feature_name in enumerate(feature_cols):
        X_permuted = X_val_tensor.clone()
        perm_idx = torch.randperm(X_permuted.shape[0])
        X_permuted[:, fi] = X_permuted[perm_idx, fi]

        with torch.no_grad():
            permuted_pred = model(X_permuted)
            permuted_loss = criterion(permuted_pred, y_val_tensor).item()

        importance = permuted_loss - baseline_loss
        importances.append(importance)

        if (fi + 1) % 5 == 0:
            print(f"  Processed {fi+1}/{len(feature_cols)} features...")

    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
    print("\n=== Top 15 Most Important Features ===")
    print(importance_df.head(15).to_string(index=False))

    # Visualization 1: Top 20 Features
    plt.figure(figsize=(12, 8))
    top_n = 20
    top_features = importance_df.head(top_n)
    colors_imp = ['#d62728' if f.startswith('pos_') else '#2ca02c' if f.startswith('side_') else '#ff7f0e' if f.startswith('role_') else '#1f77b4' for f in top_features['Feature']]
    plt.barh(range(len(top_features)), top_features['Importance'], color=colors_imp)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance (Increase in MSE Loss)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features\n(Blue=Numerical, Red=Position, Green=Side, Orange=Role)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualization 2: By Category
    num_feats = [f for f in feature_cols if not f.startswith(('pos_', 'side_', 'role_'))]
    pos_feats = [f for f in feature_cols if f.startswith('pos_')]
    side_feats = [f for f in feature_cols if f.startswith('side_')]
    role_feats = [f for f in feature_cols if f.startswith('role_')]

    num_imp = importance_df[importance_df['Feature'].isin(num_feats)]
    pos_imp = importance_df[importance_df['Feature'].isin(pos_feats)]
    side_imp = importance_df[importance_df['Feature'].isin(side_feats)]
    role_imp = importance_df[importance_df['Feature'].isin(role_feats)]

    fig2, ax2 = plt.subplots(2, 2, figsize=(16, 12))

    ax2[0, 0].barh(range(len(num_imp)), num_imp['Importance'], color='#1f77b4')
    ax2[0, 0].set_yticks(range(len(num_imp)))
    ax2[0, 0].set_yticklabels(num_imp['Feature'])
    ax2[0, 0].set_xlabel('Importance', fontsize=12)
    ax2[0, 0].set_title('Numerical Features', fontsize=14, fontweight='bold')
    ax2[0, 0].invert_yaxis()
    ax2[0, 0].grid(axis='x', alpha=0.3)

    ax2[0, 1].barh(range(len(pos_imp)), pos_imp['Importance'], color='#d62728')
    ax2[0, 1].set_yticks(range(len(pos_imp)))
    ax2[0, 1].set_yticklabels(pos_imp['Feature'])
    ax2[0, 1].set_xlabel('Importance', fontsize=12)
    ax2[0, 1].set_title('Position Features', fontsize=14, fontweight='bold')
    ax2[0, 1].invert_yaxis()
    ax2[0, 1].grid(axis='x', alpha=0.3)

    ax2[1, 0].barh(range(len(side_imp)), side_imp['Importance'], color='#2ca02c')
    ax2[1, 0].set_yticks(range(len(side_imp)))
    ax2[1, 0].set_yticklabels(side_imp['Feature'])
    ax2[1, 0].set_xlabel('Importance', fontsize=12)
    ax2[1, 0].set_title('Side Features', fontsize=14, fontweight='bold')
    ax2[1, 0].invert_yaxis()
    ax2[1, 0].grid(axis='x', alpha=0.3)

    ax2[1, 1].barh(range(len(role_imp)), role_imp['Importance'], color='#ff7f0e')
    ax2[1, 1].set_yticks(range(len(role_imp)))
    ax2[1, 1].set_yticklabels(role_imp['Feature'])
    ax2[1, 1].set_xlabel('Importance', fontsize=12)
    ax2[1, 1].set_title('Role Features', fontsize=14, fontweight='bold')
    ax2[1, 1].invert_yaxis()
    ax2[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_importance_by_category.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n=== Feature Importance Summary ===")
    print(f"Total features: {len(feature_cols)}")
    print(f"Mean importance: Numerical={num_imp['Importance'].mean():.6f}, Positions={pos_imp['Importance'].mean():.6f}, Sides={side_imp['Importance'].mean():.6f}, Roles={role_imp['Importance'].mean():.6f}")
    #####################3
    # Case study analysis
    print("\n" + "=" * 80)
    print("CASE STUDY ANALYSIS - SINGLE PLAY")
    print("=" * 80)

    case_study_game_id = 2023091700
    case_study_play_id = 989
    play_data = data[(data['game_id'] == case_study_game_id) & (data['play_id'] == case_study_play_id)]

    print(f"\nPlay: Game {case_study_game_id}, Play {case_study_play_id}")
    print(f"Number of player/frame combinations: {len(play_data)}")
    print(f"Position counts:\n{play_data['player_position'].value_counts()}")

    ball_land_x = play_data['ball_land_x'].iloc[0]
    ball_land_y = play_data['ball_land_y'].iloc[0]
    print(f"\nBall landing position: ({ball_land_x:.2f}, {ball_land_y:.2f})")

    # Prepare predictions
    play_numerical = play_data[numerical_features].astype(np.float64)
    play_position_dummies = pd.get_dummies(play_data['player_position'], prefix='pos')
    play_side_dummies = pd.get_dummies(play_data['player_side'], prefix='side')
    play_role_dummies = pd.get_dummies(play_data['player_role'], prefix='role')

    for col in position_dummies.columns:
        if col not in play_position_dummies.columns:
            play_position_dummies[col] = 0
    for col in side_dummies.columns:
        if col not in play_side_dummies.columns:
            play_side_dummies[col] = 0
    for col in role_dummies.columns:
        if col not in play_role_dummies.columns:
            play_role_dummies[col] = 0

    play_position_dummies = play_position_dummies[position_dummies.columns]
    play_side_dummies = play_side_dummies[side_dummies.columns]
    play_role_dummies = play_role_dummies[role_dummies.columns]

    play_combined = pd.concat([
        play_numerical.reset_index(drop=True),
        play_position_dummies.reset_index(drop=True),
        play_side_dummies.reset_index(drop=True),
        play_role_dummies.reset_index(drop=True)
    ], axis=1)

    X_play = play_combined[feature_cols].values.astype(np.float64)
    X_play_scaled = scaler.transform(X_play)
    X_play_tensor = torch.FloatTensor(X_play_scaled).to(device)

    model.eval()
    with torch.no_grad():
        play_predictions = model(X_play_tensor).cpu().numpy()

    play_actual = play_data[["x_pos", "y_pos"]].values
    play_errors = play_predictions - play_actual
    play_mae = mean_absolute_error(play_actual, play_predictions)

    print(f"\n=== Case Study Play Predictions ===")
    print(f"Mean Absolute Error: {play_mae:.4f} yards")
    print(f"Euclidean error: {np.mean(np.sqrt(np.sum(play_errors**2, axis=1))):.4f} yards")
    print(f"\nErrors by position:")
    for pos in play_data['player_position'].unique():
        pos_mask = play_data['player_position'] == pos
        pos_mae = mean_absolute_error(play_actual[pos_mask], play_predictions[pos_mask])
        print(f"  {pos}: {pos_mae:.4f} yards")

    # Visualization 1: Pre-snap vs Post-snap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    ax1.scatter(play_data['x_pre'], play_data['y_pre'], s=100, alpha=0.7, c='blue', edgecolors='white', linewidths=1)
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 53.3)
    ax1.set_xlabel("X Position (yards)")
    ax1.set_ylabel("Y Position (yards)")
    ax1.set_title("Pre-snap Positions")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(play_data['x_pos'], play_data['y_pos'], s=100, alpha=0.7, c='blue', edgecolors='white', linewidths=1, label='Actual', marker='o')
    ax2.scatter(play_predictions[:, 0], play_predictions[:, 1], s=100, alpha=0.7, c='orange', edgecolors='white', linewidths=1, label='Predicted', marker='^')
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 53.3)
    ax2.set_xlabel("X Position (yards)")
    ax2.set_ylabel("Y Position (yards)")
    ax2.set_title("Post-snap: Actual vs Predicted")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('case_study_play.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualization 2: Movement vectors on field
    fig2, ax = plt.subplots(figsize=(16, 8.5))
    try:
        field_img = plt.imread('Projects/NFL/nfl-big-data-bowl-2026-prediction/field.png')
        ax.imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0, alpha=0.3)
    except:
        ax.set_facecolor('#90EE90')
        ax.text(60, 26.65, 'Football Field', ha='center', va='center', fontsize=20, alpha=0.3)

    ax.scatter(play_data['x_pre'], play_data['y_pre'], s=150, marker='o', c='darkblue', edgecolors='white', linewidths=2, label='Pre-snap', zorder=5, alpha=0.8)
    ax.scatter(play_data['x_pos'], play_data['y_pos'], s=150, marker='s', c='blue', edgecolors='white', linewidths=2, label='Actual Post-snap', zorder=5, alpha=0.8)
    ax.scatter(play_predictions[:, 0], play_predictions[:, 1], s=150, marker='^', c='orange', edgecolors='white', linewidths=2, label='Predicted Post-snap', zorder=5, alpha=0.8)

    for i in range(len(play_data)):
        x_start, y_start = play_data['x_pre'].iloc[i], play_data['y_pre'].iloc[i]
        x_end_actual, y_end_actual = play_data['x_pos'].iloc[i], play_data['y_pos'].iloc[i]
        ax.annotate('', xy=(x_end_actual, y_end_actual), xytext=(x_start, y_start),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.6), zorder=3)

    for i in range(len(play_data)):
        x_start, y_start = play_data['x_pre'].iloc[i], play_data['y_pre'].iloc[i]
        x_end_pred, y_end_pred = play_predictions[i, 0], play_predictions[i, 1]
        ax.annotate('', xy=(x_end_pred, y_end_pred), xytext=(x_start, y_start),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='orange', alpha=0.7, linestyle='dashed'), zorder=2)

    ax.scatter([ball_land_x], [ball_land_y], s=300, marker='*', c='red', edgecolors='darkred', linewidths=2, label='Ball Landing', zorder=6, alpha=0.9)
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 58.3)
    ax.set_xlabel("X Position (yards)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Position (yards)", fontsize=14, fontweight='bold')
    ax.set_title("Case Study Play: Movement Vectors (Blue=Actual, Orange=Predicted)", fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, markerscale=1.5)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('case_study_movement.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualization 3: Error heatmap
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    error_magnitudes = np.sqrt(np.sum((play_actual - play_predictions)**2, axis=1))
    positions = play_data['player_position'].values
    unique_positions = np.unique(positions)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_positions)))
    for i, pos in enumerate(unique_positions):
        mask = positions == pos
        ax3.scatter(play_data['x_pos'].values[mask], play_data['y_pos'].values[mask], 
                   s=error_magnitudes[mask]*100 + 50, alpha=0.7, c=[colors[i]], edgecolors='white', linewidths=1.5, label=pos)
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 53.3)
    ax3.set_xlabel("X Position (yards)")
    ax3.set_ylabel("Y Position (yards)")
    ax3.set_title("Error Magnitude by Position (size = error)")
    ax3.legend(loc='upper left', ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('case_study_error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualization 4: Distance from ball
    actual_dist_from_ball = np.sqrt((play_data['x_pos'].values - ball_land_x)**2 + (play_data['y_pos'].values - ball_land_y)**2)
    pred_dist_from_ball = np.sqrt((play_predictions[:, 0] - ball_land_x)**2 + (play_predictions[:, 1] - ball_land_y)**2)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    ax4a.scatter(actual_dist_from_ball, error_magnitudes, s=100, alpha=0.7, c=play_data['player_position'].astype('category').cat.codes, cmap='tab20')
    ax4a.set_xlabel("Actual Distance from Ball Landing (yards)")
    ax4a.set_ylabel("Prediction Error (yards)")
    ax4a.set_title("Prediction Error vs Distance from Ball")
    ax4a.grid(True, alpha=0.3)

    ax4b.scatter(actual_dist_from_ball, pred_dist_from_ball, s=100, alpha=0.7)
    ax4b.plot([actual_dist_from_ball.min(), actual_dist_from_ball.max()], [actual_dist_from_ball.min(), actual_dist_from_ball.max()], 'r--', linewidth=2, label='Perfect')
    ax4b.set_xlabel("Actual Distance from Ball (yards)")
    ax4b.set_ylabel("Predicted Distance from Ball (yards)")
    ax4b.set_title("Distance from Ball: Actual vs Predicted")
    ax4b.legend()
    ax4b.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('case_study_ball_distance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== Distance from Ball Analysis ===")
    print(f"Average actual distance from ball: {actual_dist_from_ball.mean():.2f} yards")
    print(f"Average predicted distance from ball: {pred_dist_from_ball.mean():.2f} yards")
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), 'best_model.pth')
    print("✅ Model saved!")
    return


if __name__ == "__main__":
    app.run()
