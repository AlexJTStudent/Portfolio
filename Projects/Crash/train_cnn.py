"""
Car Crash Detection - CNN Training Script
Uses EfficientNetB0 + Grad-CAM for classification and heatmap generation
"""

import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
IMG_SIZE = 224  # EfficientNet standard input size
BATCH_SIZE = 10
EPOCHS = 25
LEARNING_RATE = 0.003

# Data paths
DATA_DIR_NOT_CRASHED = r'C:\Users\Alex\Documents\GitHub\Portfolio\Projects\Crash\Data\AccidentDetection\1'
DATA_DIR_CRASHED = r'C:\Users\Alex\Documents\GitHub\Portfolio\Projects\Crash\Data\AccidentDetection\2'
OUTPUT_DIR = 'models_nn'

print("=" * 60)
print("Car Crash Detection - Neural Network Training")
print("=" * 60)
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 60)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_image(image_path, label):
    """Load and preprocess a single image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_array, label
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None

def load_dataset():
    """Load all images from both classes"""
    print("\nLoading dataset...")
    
    X = []
    y = []
    
    # Load not-crashed images (class 0)
    print(f"Loading not-crashed images from: {DATA_DIR_NOT_CRASHED}")
    count_not_crashed = 0
    for img_name in os.listdir(DATA_DIR_NOT_CRASHED):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(DATA_DIR_NOT_CRASHED, img_name)
            img_array, label = load_and_preprocess_image(img_path, 0)
            if img_array is not None:
                X.append(img_array)
                y.append(label)
                count_not_crashed += 1
                if count_not_crashed % 100 == 0:
                    print(f"  Loaded {count_not_crashed} not-crashed images...")
    
    print(f"✓ Total not-crashed images: {count_not_crashed}")
    
    # Load crashed images (class 1)
    print(f"Loading crashed images from: {DATA_DIR_CRASHED}")
    count_crashed = 0
    for img_name in os.listdir(DATA_DIR_CRASHED):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(DATA_DIR_CRASHED, img_name)
            img_array, label = load_and_preprocess_image(img_path, 1)
            if img_array is not None:
                X.append(img_array)
                y.append(label)
                count_crashed += 1
                if count_crashed % 100 == 0:
                    print(f"  Loaded {count_crashed} crashed images...")
    
    print(f"✓ Total crashed images: {count_crashed}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total images: {len(X)}")
    print(f"Image shape: {X.shape}")
    print(f"Class distribution: Not-crashed={count_not_crashed}, Crashed={count_crashed}")
    
    return X, y

def create_model():
    """Create EfficientNetB0 model for binary classification"""
    print("\nCreating model architecture...")
    
    # Build model from scratch without pre-trained weights first
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Load pre-trained EfficientNetB0 with updated method
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,  # Load weights separately
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Try to load ImageNet weights if available
    try:
        print("Attempting to load ImageNet weights...")
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            input_tensor=inputs
        )
        print("✓ ImageNet weights loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load ImageNet weights: {e}")
        print("Training from scratch (will take longer but should still work)")
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            input_tensor=inputs
        )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Base model
    x = base_model.output
    
    # Global pooling and classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(base_model.input, outputs, name='car_crash_detector')
    
    print("✓ Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    return model, base_model

def plot_training_history(history, output_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {output_path}")
    plt.close()

def main():
    """Main training pipeline"""
    start_time = datetime.now()
    
    # Load data
    X, y = load_dataset()
    
    # Split data
    print("\nSplitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create model
    model, base_model = create_model()
    
    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train Phase 1: Train only top layers
    print("\n" + "=" * 60)
    print("PHASE 1: Training classification head (frozen base)")
    print("=" * 60)
    
    history_phase1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=15,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Train Phase 2: Fine-tune entire model
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning entire model")
    print("=" * 60)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    history_phase2 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS - 15,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine history
    history = type('obj', (object,), {
        'history': {
            'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
            'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
            'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
            'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
        }
    })()
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")
    
    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, 'car_crash_model_final.keras')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Save training history plot
    plot_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Save model info
    info_path = os.path.join(OUTPUT_DIR, 'model_info.txt')
    with open(info_path, 'w') as f:
        f.write("Car Crash Detection Model - Training Info\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: EfficientNetB0\n")
        f.write(f"Input Size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Total Parameters: {model.count_params():,}\n")
        f.write(f"Training Images: {len(X_train)}\n")
        f.write(f"Validation Images: {len(X_val)}\n")
        f.write(f"Test Images: {len(X_test)}\n\n")
        f.write("Test Results:\n")
        f.write(f"  Accuracy: {test_results[1]:.4f}\n")
        f.write(f"  Precision: {test_results[2]:.4f}\n")
        f.write(f"  Recall: {test_results[3]:.4f}\n")
        f.write(f"  Loss: {test_results[0]:.4f}\n")
    
    print(f"✓ Model info saved to: {info_path}")
    
    # Training complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total training time: {duration}")
    print(f"Models saved in: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Check training_history.png to verify training went well")
    print("2. Use 'best_model.keras' for deployment")
    print("=" * 60)

if __name__ == '__main__':
    # Check TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Run training
    main()