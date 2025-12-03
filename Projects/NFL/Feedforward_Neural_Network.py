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
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    return


@app.cell
def _(torch):
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(pd):
    # ========== 1. Load Data ==========
    print("Loading data...")
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

    test = pd.read_csv("Projects/NFL/nfl-big-data-bowl-2026-prediction/test.csv")
    return test, wo1_pos, wo1_pre


@app.cell
def _(wo1_pos, wo1_pre):
    print(wo1_pre.info())
    print(wo1_pos.info())
    return


@app.cell
def _(
    DataLoader,
    StandardScaler,
    TensorDataset,
    device,
    nn,
    torch,
    train_test_split,
    wo1_pos,
    wo1_pre,
):
    # ========== 2. Merge and Prepare Data ==========
    print("Merging data...")
    data = wo1_pre.merge(
        wo1_pos,
        on=["game_id", "play_id", "nfl_id", "frame_id"],
        suffixes=("_pre", "_pos")
    )

    feature_cols = [
        "x_pre", "y_pre", "s", "a", "dir", "o",
        "absolute_yardline_number", "ball_land_x", "ball_land_y"
    ]

    X = data[feature_cols].values
    y = data[["x_pos", "y_pos"]].values

    # ========== 3. Train/Val Split ==========
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ========== 4. Scale Features (Important for Neural Networks!) ==========
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ========== 5. Convert to PyTorch Tensors ==========
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # ========== 6. Define Neural Network Model ==========
    class PositionPredictor(nn.Module):
        def __init__(self, input_dim):
            super(PositionPredictor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32, 2)  # Output: x_pos and y_pos
            )

        def forward(self, x):
            return self.network(x)
    return (
        PositionPredictor,
        X_val,
        X_val_tensor,
        data,
        feature_cols,
        scaler,
        train_loader,
        y_val,
        y_val_tensor,
    )


@app.cell
def _(PositionPredictor, device, feature_cols, nn, optim):
    # ========== 7. Initialize Model, Loss, Optimizer ==========
    input_dim = len(feature_cols)
    model = PositionPredictor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
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
    # ========== 8. Training Loop ==========
    print("\nTraining model...")
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
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

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
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
    # ========== 9. Evaluate Model ==========
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor).cpu().numpy()

    x_pred_val = predictions[:, 0]
    y_pred_val = predictions[:, 1]

    # Calculate metrics
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
def _(plt, train_losses, val_losses, x_pred_val, y_pred_val, y_val):
    # ========== 10. Visualizations ==========
    print("\nGenerating visualizations...")

    # Training history
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

    # Predicted vs Actual X
    plt.figure(figsize=(10, 5))
    plt.scatter(y_val[:, 0], x_pred_val, alpha=0.3, s=10)
    plt.plot([y_val[:, 0].min(), y_val[:, 0].max()],
             [y_val[:, 0].min(), y_val[:, 0].max()],
             color='red', linestyle='--', linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual X Position")
    plt.ylabel("Predicted X Position")
    plt.title("Predicted vs Actual X Positions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('x_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Predicted vs Actual Y
    plt.figure(figsize=(10, 5))
    plt.scatter(y_val[:, 1], y_pred_val, alpha=0.3, s=10)
    plt.plot([y_val[:, 1].min(), y_val[:, 1].max()],
             [y_val[:, 1].min(), y_val[:, 1].max()],
             color='red', linestyle='--', linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual Y Position")
    plt.ylabel("Predicted Y Position")
    plt.title("Predicted vs Actual Y Positions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('y_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(np, plt, x_pred_val, y_pred_val, y_val):
    # 2D position comparison with football field background
    plt.figure(figsize=(12, 6.4))  # Football field aspect ratio (120 yards x 53.3 yards)

    # Load and display the field image
    field_img = plt.imread('Projects/NFL/nfl-big-data-bowl-2026-prediction/field.png')
    plt.imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0)

    # Sample 1000 points for cleaner visualization
    sample_indices = np.random.choice(len(y_val), size=min(1000, len(y_val)), replace=False)

    # Plot actual positions
    plt.scatter(y_val[sample_indices, 0], y_val[sample_indices, 1], 
                alpha=0.6, s=50, label="Actual", color='blue', edgecolors='white', linewidths=0.5, zorder=2)

    # Plot predicted positions
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
    return (field_img,)


@app.cell
def _():
    return


@app.cell
def _(device, feature_cols, model, scaler, test, torch, wo1_pre):
    # ========== 11. Prepare Test Predictions ==========
    print("\nPreparing test predictions...")
    wo1_pre_renamed = wo1_pre.rename(columns={"x": "x_pre", "y": "y_pre"})

    test_data = test.merge(
        wo1_pre_renamed,
        on=["game_id", "play_id", "nfl_id", "frame_id"],
        how="left"
    )

    X_test = test_data[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy()

    test_data["x_pred"] = test_predictions[:, 0]
    test_data["y_pred"] = test_predictions[:, 1]

    # ========== 12. Export Submission ==========
    submission = test_data[["game_id", "play_id", "nfl_id", "frame_id", "x_pred", "y_pred"]]
    submission.to_csv("submission.csv", index=False)

    print("✅ Predictions saved to submission.csv")
    print(f"✅ Submission shape: {submission.shape}")
    print("\nDone!")
    return


@app.cell
def _(
    X_val,
    data,
    device,
    feature_cols,
    field_img,
    model,
    np,
    plt,
    scaler,
    torch,
):
    # ========== Case Study: Single Play Visualization ==========
    print("\n=== Case Study: Single Play ===")

    # Select a random play from validation set
    # Get unique plays
    val_indices = X_val.shape[0]
    play_sample_idx = np.random.randint(0, val_indices)

    # Get the play details from original data
    sample_game_id = data.iloc[play_sample_idx]['game_id']
    sample_play_id = data.iloc[play_sample_idx]['play_id']

    print(f"Game ID: {sample_game_id}")
    print(f"Play ID: {sample_play_id}")

    # Get all frames for this specific play
    play_data = data[(data['game_id'] == sample_game_id) & (data['play_id'] == sample_play_id)].copy()

    print(f"Number of players/frames in this play: {len(play_data)}")

    # Get ball position (should be same for all players in a frame)
    ball_land_x = play_data['ball_land_x'].iloc[0]
    ball_land_y = play_data['ball_land_y'].iloc[0]

    print(f"\nBall landing position: X={ball_land_x:.2f}, Y={ball_land_y:.2f}")

    # Prepare features for this play
    X_play = play_data[feature_cols].values
    X_play_scaled = scaler.transform(X_play)
    X_play_tensor = torch.FloatTensor(X_play_scaled).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        play_predictions = model(X_play_tensor).cpu().numpy()

    play_data['x_pred'] = play_predictions[:, 0]
    play_data['y_pred'] = play_predictions[:, 1]

    # Calculate errors for this play
    play_data['error_x'] = abs(play_data['x_pos'] - play_data['x_pred'])
    play_data['error_y'] = abs(play_data['y_pos'] - play_data['y_pred'])
    play_data['error_euclidean'] = np.sqrt(play_data['error_x']**2 + play_data['error_y']**2)

    print(f"\nPlay Statistics:")
    print(f"Mean X Error: {play_data['error_x'].mean():.2f} yards")
    print(f"Mean Y Error: {play_data['error_y'].mean():.2f} yards")
    print(f"Mean Euclidean Error: {play_data['error_euclidean'].mean():.2f} yards")
    print(f"Max Euclidean Error: {play_data['error_euclidean'].max():.2f} yards")

    # ========== Visualization 1: Pre-snap and Post-snap Positions ==========
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.4))

    # Pre-snap positions (left plot)
    axes[0].imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0)
    axes[0].scatter(play_data['x_pre'], play_data['y_pre'], 
                    s=200, c='green', alpha=0.8, edgecolors='white', 
                    linewidths=2, zorder=2, label='Pre-snap Players')
    # Ball at line of scrimmage (approximate as average x_pre)
    line_of_scrimmage_x = play_data['x_pre'].mean()
    axes[0].scatter(line_of_scrimmage_x, 26.65, s=250, c='brown', 
                    marker='o', edgecolors='white', linewidths=2, 
                    zorder=3, label='Ball (Pre-snap)')
    # Ball landing position
    axes[0].scatter(ball_land_x, ball_land_y, s=250, c='red', 
                    marker='X', edgecolors='white', linewidths=2, 
                    zorder=3, label='Ball Landing')
    axes[0].set_xlabel("X Position (yards)", fontsize=12)
    axes[0].set_ylabel("Y Position (yards)", fontsize=12)
    axes[0].set_title("Pre-Snap Positions", fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=12, markerscale=1.2)
    axes[0].set_xlim(0, 120)
    axes[0].set_ylim(0, 53.3)

    # Post-snap: Actual vs Predicted (right plot)
    axes[1].imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0)
    axes[1].scatter(play_data['x_pos'], play_data['y_pos'], 
                    s=200, c='blue', alpha=0.7, edgecolors='white', 
                    linewidths=2, zorder=2, label='Actual Post-snap')
    axes[1].scatter(play_data['x_pred'], play_data['y_pred'], 
                    s=200, c='orange', alpha=0.7, edgecolors='white', 
                    linewidths=2, zorder=2, label='Predicted Post-snap')
    # Ball landing position
    axes[1].scatter(ball_land_x, ball_land_y, s=300, c='red', 
                    marker='X', edgecolors='white', linewidths=2, 
                    zorder=4, label='Ball Landing')

    # Draw lines connecting actual to predicted
    for idx in play_data.index:
        axes[1].plot([play_data.loc[idx, 'x_pos'], play_data.loc[idx, 'x_pred']], 
                     [play_data.loc[idx, 'y_pos'], play_data.loc[idx, 'y_pred']], 
                     'r--', alpha=0.4, linewidth=1, zorder=1)

    axes[1].set_xlabel("X Position (yards)", fontsize=12)
    axes[1].set_ylabel("Y Position (yards)", fontsize=12)
    axes[1].set_title("Post-Snap: Actual vs Predicted", fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=12, markerscale=1.2)
    axes[1].set_xlim(0, 120)
    axes[1].set_ylim(0, 53.3)

    plt.tight_layout()
    plt.savefig('case_study_play.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ========== Visualization 2: Movement Vectors ==========
    plt.figure(figsize=(12, 6.4))
    plt.imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0)

    # Pre-snap positions
    plt.scatter(play_data['x_pre'], play_data['y_pre'], 
                s=150, c='green', alpha=0.8, edgecolors='white', 
                linewidths=2, zorder=2, label='Pre-snap', marker='o')

    # Ball trajectory (line of scrimmage to landing)
    line_of_scrimmage_x = play_data['x_pre'].mean()
    plt.plot([line_of_scrimmage_x, ball_land_x], [26.65, ball_land_y], 
             'r-', linewidth=4, alpha=0.7, zorder=3, label='Ball trajectory')
    plt.scatter(line_of_scrimmage_x, 26.65, s=200, c='brown', 
                marker='o', edgecolors='white', linewidths=2, zorder=4)
    plt.scatter(ball_land_x, ball_land_y, s=300, c='red', 
                marker='X', edgecolors='white', linewidths=2, zorder=4)

    # Actual movement (blue arrows)
    for idx in play_data.index:
        plt.arrow(play_data.loc[idx, 'x_pre'], play_data.loc[idx, 'y_pre'],
                  play_data.loc[idx, 'x_pos'] - play_data.loc[idx, 'x_pre'],
                  play_data.loc[idx, 'y_pos'] - play_data.loc[idx, 'y_pre'],
                  head_width=1.5, head_length=1, fc='blue', ec='blue', 
                  alpha=0.6, linewidth=2, zorder=1, label='Actual movement' if idx == play_data.index[0] else '')

    # Predicted movement (orange arrows)
    for idx in play_data.index:
        plt.arrow(play_data.loc[idx, 'x_pre'], play_data.loc[idx, 'y_pre'],
                  play_data.loc[idx, 'x_pred'] - play_data.loc[idx, 'x_pre'],
                  play_data.loc[idx, 'y_pred'] - play_data.loc[idx, 'y_pre'],
                  head_width=1.5, head_length=1, fc='orange', ec='orange', 
                  alpha=0.4, linewidth=1.5, zorder=1, linestyle='--',
                  label='Predicted movement' if idx == play_data.index[0] else '')

    plt.xlabel("X Position (yards)", fontsize=12)
    plt.ylabel("Y Position (yards)", fontsize=12)
    plt.title("Player Movement: Actual vs Predicted (with Ball Trajectory)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.savefig('case_study_movement.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ========== Visualization 3: Error Heatmap by Position ==========
    plt.figure(figsize=(12, 6.4))
    plt.imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0)

    # Color code by error magnitude
    scatter = plt.scatter(play_data['x_pos'], play_data['y_pos'], 
                          c=play_data['error_euclidean'], 
                          s=300, cmap='YlOrRd', alpha=0.8, 
                          edgecolors='white', linewidths=2, zorder=2)

    # Ball landing position
    plt.scatter(ball_land_x, ball_land_y, s=350, c='red', 
                marker='X', edgecolors='black', linewidths=3, 
                zorder=4, label='Ball Landing')

    # Add colorbar
    cbar = plt.colorbar(scatter, label='Prediction Error (yards)')
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel("X Position (yards)", fontsize=12)
    plt.ylabel("Y Position (yards)", fontsize=12)
    plt.title("Prediction Error by Player Position (with Ball Landing)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12, markerscale=1.2)
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.savefig('case_study_error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ========== Visualization 4: Distance from Ball Analysis ==========
    plt.figure(figsize=(12, 6.4))
    plt.imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto', zorder=0)

    # Calculate distance from ball for each player
    play_data['dist_from_ball_actual'] = np.sqrt(
        (play_data['x_pos'] - ball_land_x)**2 + 
        (play_data['y_pos'] - ball_land_y)**2
    )
    play_data['dist_from_ball_pred'] = np.sqrt(
        (play_data['x_pred'] - ball_land_x)**2 + 
        (play_data['y_pred'] - ball_land_y)**2
    )

    # Plot players color-coded by distance from ball (actual)
    scatter = plt.scatter(play_data['x_pos'], play_data['y_pos'], 
                          c=play_data['dist_from_ball_actual'], 
                          s=300, cmap='viridis', alpha=0.8, 
                          edgecolors='white', linewidths=2, zorder=2)

    # Ball landing position
    plt.scatter(ball_land_x, ball_land_y, s=400, c='red', 
                marker='*', edgecolors='white', linewidths=3, 
                zorder=4, label='Ball Landing')

    # Draw circles around ball at 5, 10, 15 yard intervals
    for radius in [5, 10, 15]:
        circle = plt.Circle((ball_land_x, ball_land_y), radius, 
                            color='red', fill=False, linestyle='--', 
                            linewidth=2, alpha=0.5, zorder=1)
        plt.gca().add_patch(circle)

    # Add colorbar
    cbar = plt.colorbar(scatter, label='Distance from Ball (yards)')
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel("X Position (yards)", fontsize=12)
    plt.ylabel("Y Position (yards)", fontsize=12)
    plt.title("Player Distance from Ball Landing Position", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12, markerscale=1.2)
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.savefig('case_study_ball_distance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ========== Print detailed player-by-player breakdown ==========
    print("\n=== Player-by-Player Breakdown ===")
    play_summary = play_data[['nfl_id', 'x_pre', 'y_pre', 'x_pos', 'y_pos', 
                               'x_pred', 'y_pred', 'error_euclidean', 
                               'dist_from_ball_actual', 'dist_from_ball_pred']].copy()
    play_summary = play_summary.sort_values('error_euclidean', ascending=False)
    print(play_summary.to_string(index=False))

    print(f"\n=== Ball and Player Relationship ===")
    print(f"Ball Landing Position: ({ball_land_x:.2f}, {ball_land_y:.2f})")
    print(f"Average player distance from ball (actual): {play_data['dist_from_ball_actual'].mean():.2f} yards")
    print(f"Average player distance from ball (predicted): {play_data['dist_from_ball_pred'].mean():.2f} yards")
    print(f"Closest player to ball (actual): {play_data['dist_from_ball_actual'].min():.2f} yards")
    print(f"Closest player to ball (predicted): {play_data['dist_from_ball_pred'].min():.2f} yards")
    return


if __name__ == "__main__":
    app.run()


@app.cell
def _(
    X_train_tensor,
    X_val_tensor,
    criterion,
    feature_cols,
    model,
    np,
    pd,
    plt,
    torch,
    y_train_tensor,
    y_val_tensor,
):
    # ========== Feature Importance Analysis ==========
    print("\n=== Feature Importance Analysis ===")
    
    # Method 1: Permutation Importance
    print("\nCalculating permutation importance...")
    
    model.eval()
    
    # Get baseline validation loss
    with torch.no_grad():
        baseline_pred = model(X_val_tensor)
        baseline_loss = criterion(baseline_pred, y_val_tensor).item()
    
    print(f"Baseline validation loss: {baseline_loss:.4f}")
    
    # Calculate importance for each feature
    importances = []
    
    for i, feature_name in enumerate(feature_cols):
        # Create a copy of validation data
        X_permuted = X_val_tensor.clone()
        
        # Permute (shuffle) the i-th feature
        perm_idx = torch.randperm(X_permuted.shape[0])
        X_permuted[:, i] = X_permuted[perm_idx, i]
        
        # Calculate loss with permuted feature
        with torch.no_grad():
            permuted_pred = model(X_permuted)
            permuted_loss = criterion(permuted_pred, y_val_tensor).item()
        
        # Importance = increase in loss when feature is shuffled
        importance = permuted_loss - baseline_loss
        importances.append(importance)
        
        if i % 5 == 0:
            print(f"Processed {i+1}/{len(feature_cols)} features...")
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n=== Top 15 Most Important Features ===")
    print(importance_df.head(15).to_string(index=False))
    
    print("\n=== Bottom 10 Least Important Features ===")
    print(importance_df.tail(10).to_string(index=False))
    
    # ========== Visualization 1: Top Features Bar Plot ==========
    plt.figure(figsize=(12, 8))
    top_n = 20
    top_features = importance_df.head(top_n)
    
    colors = ['#d62728' if f.startswith('pos_') else '#1f77b4' for f in top_features['Feature']]
    
    plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance (Increase in MSE Loss)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features\n(Red = Position, Blue = Numerical)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== Visualization 2: Feature Categories ==========
    # Separate numerical and position features
    numerical_features = [f for f in feature_cols if not f.startswith('pos_')]
    position_features = [f for f in feature_cols if f.startswith('pos_')]
    
    numerical_importance = importance_df[importance_df['Feature'].isin(numerical_features)]
    position_importance = importance_df[importance_df['Feature'].isin(position_features)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Numerical features
    axes[0].barh(range(len(numerical_importance)), numerical_importance['Importance'], color='#1f77b4')
    axes[0].set_yticks(range(len(numerical_importance)))
    axes[0].set_yticklabels(numerical_importance['Feature'])
    axes[0].set_xlabel('Importance (Increase in MSE Loss)', fontsize=12)
    axes[0].set_title('Numerical Features Importance', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Position features
    axes[1].barh(range(len(position_importance)), position_importance['Importance'], color='#d62728')
    axes[1].set_yticks(range(len(position_importance)))
    axes[1].set_yticklabels(position_importance['Feature'])
    axes[1].set_xlabel('Importance (Increase in MSE Loss)', fontsize=12)
    axes[1].set_title('Player Position Features Importance', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_by_category.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== Summary Statistics ==========
    print("\n=== Feature Importance Summary ===")
    print(f"Total features: {len(feature_cols)}")
    print(f"  - Numerical: {len(numerical_features)}")
    print(f"  - Positions: {len(position_features)}")
    print(f"\nMean importance:")
    print(f"  - Numerical: {numerical_importance['Importance'].mean():.6f}")
    print(f"  - Positions: {position_importance['Importance'].mean():.6f}")
    print(f"\nTop 3 overall:")
    for idx, row in importance_df.head(3).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.6f}")
    
    # ========== Method 2: Weight Magnitude Analysis ==========
    print("\n\n=== Weight Magnitude Analysis ===")
    print("Analyzing first layer weights to see which features have strongest connections...")
    
    # Get weights from first layer
    first_layer_weights = model.network[0].weight.data.cpu().numpy()  # Shape: (256, num_features)
    
    # Calculate average absolute weight for each feature
    weight_magnitudes = np.abs(first_layer_weights).mean(axis=0)
    
    weight_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Weight_Magnitude': weight_magnitudes
    }).sort_values('Weight_Magnitude', ascending=False)
    
    print("\n=== Top 15 Features by Weight Magnitude ===")
    print(weight_importance_df.head(15).to_string(index=False))
    
    # ========== Visualization 3: Weight Magnitude ==========
    plt.figure(figsize=(12, 8))
    top_weights = weight_importance_df.head(20)
    colors_w = ['#d62728' if f.startswith('pos_') else '#1f77b4' for f in top_weights['Feature']]
    
    plt.barh(range(len(top_weights)), top_weights['Weight_Magnitude'], color=colors_w)
    plt.yticks(range(len(top_weights)), top_weights['Feature'])
    plt.xlabel('Average Absolute Weight Magnitude', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 20 Features by First Layer Weight Magnitude\n(Red = Position, Blue = Numerical)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== Comparison Plot ==========
    print("\n\n=== Comparing Importance Methods ===")
    
    # Normalize both importance measures to 0-1 scale
    importance_df['Importance_Normalized'] = (
        (importance_df['Importance'] - importance_df['Importance'].min()) / 
        (importance_df['Importance'].max() - importance_df['Importance'].min())
    )
    
    weight_importance_df['Weight_Normalized'] = (
        (weight_importance_df['Weight_Magnitude'] - weight_importance_df['Weight_Magnitude'].min()) / 
        (weight_importance_df['Weight_Magnitude'].max() - weight_importance_df['Weight_Magnitude'].min())
    )
    
    # Merge the two methods
    comparison_df = importance_df[['Feature', 'Importance_Normalized']].merge(
        weight_importance_df[['Feature', 'Weight_Normalized']], 
        on='Feature'
    )
    comparison_df['Average_Importance'] = (
        comparison_df['Importance_Normalized'] + comparison_df['Weight_Normalized']
    ) / 2
    comparison_df = comparison_df.sort_values('Average_Importance', ascending=False)
    
    print("\n=== Top 15 Features (Combined Methods) ===")
    print(comparison_df.head(15)[['Feature', 'Average_Importance']].to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    top_combined = comparison_df.head(20)
    
    x = np.arange(len(top_combined))
    width = 0.35
    
    colors_comb = ['red' if f.startswith('pos_') else 'blue' for f in top_combined['Feature']]
    
    plt.barh(x - width/2, top_combined['Importance_Normalized'], width, 
             label='Permutation Importance', alpha=0.8, color='steelblue')
    plt.barh(x + width/2, top_combined['Weight_Normalized'], width, 
             label='Weight Magnitude', alpha=0.8, color='coral')
    
    plt.yticks(x, top_combined['Feature'])
    plt.xlabel('Normalized Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance: Comparison of Methods\n(Top 20 Features)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return importance dataframes for further analysis
    return importance_df, weight_importance_df, comparison_df