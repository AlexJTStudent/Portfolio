import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo
    import os
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    import matplotlib.pyplot as plt
    return ImageDataGenerator, MobileNetV2, keras, layers, os, plt, tf


@app.cell
def _(os, tf):
    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    DATA_DIR = 'Projects/Crash/Data'

    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    # Create output directory for model
    os.makedirs('crash/models', exist_ok=True)

    return BATCH_SIZE, DATA_DIR, EPOCHS, IMG_SIZE


@app.cell
def _(BATCH_SIZE, DATA_DIR, IMG_SIZE, ImageDataGenerator):
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 80% train, 20% validation
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Binary classification
        subset='training',
        shuffle=True
    )

    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )

    return train_generator, validation_generator


@app.cell
def _(train_generator, validation_generator):
    print("\nClass mapping:", train_generator.class_indices)
    print("Number of training samples:", train_generator.samples)
    print("Number of validation samples:", validation_generator.samples)
    return


@app.cell
def _(IMG_SIZE, MobileNetV2, keras, layers):
    # Build model using transfer learning with MobileNetV2
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    return (model,)


@app.cell
def _(keras, model):
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    print("\nModel Summary:")
    model.summary()
    return


@app.cell
def _(EPOCHS, keras, model, train_generator, validation_generator):
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate model
    print("\n" + "="*50)
    print("Evaluating model...")
    print("="*50)

    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(validation_generator)
    print(f"\nValidation Results:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    return (history,)


@app.cell
def _(model):
    # Save model
    model_path = 'crash/models/car_crash_model.h5'
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")

    return (model_path,)


@app.cell
def _(history, model_path, plt):
    # Plot training history
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('crash/models/training_history.png')
    print(f"✓ Training plots saved to: crash/models/training_history.png")

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"\nNext steps:")
    print(f"1. Check the training plots in crash/models/training_history.png")
    print(f"2. Upload {model_path} to AWS S3")
    print(f"3. Update Lambda function to use this model")
    return


if __name__ == "__main__":
    app.run()
