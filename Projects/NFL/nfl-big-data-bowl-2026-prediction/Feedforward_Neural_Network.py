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
        X_val_tensor,
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
    # 2D position comparison
    plt.figure(figsize=(10, 10))
    # Sample 1000 points for cleaner visualization
    sample_indices = np.random.choice(len(y_val), size=min(1000, len(y_val)), replace=False)
    plt.scatter(y_val[sample_indices, 0], y_val[sample_indices, 1], 
                alpha=0.4, s=30, label="Actual", color='blue')
    plt.scatter(x_pred_val[sample_indices], y_pred_val[sample_indices], 
                alpha=0.4, s=30, label="Predicted", color='orange')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Actual vs Predicted Player Positions (Sample)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig('position_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

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


if __name__ == "__main__":
    app.run()
