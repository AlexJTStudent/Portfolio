import tensorflow as tf
import numpy as np

# Load model
print("Loading model...")
model = tf.keras.models.load_model('models/car_crash_model.h5')

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('models/car_crash_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✓ Conversion complete!")
print(f"TFLite model size: {len(tflite_model) / (1024*1024):.2f} MB")
