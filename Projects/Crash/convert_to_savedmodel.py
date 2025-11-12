import tensorflow as tf
import os
import shutil
import tarfile

# Load the .h5 model
model = tf.keras.models.load_model('models/car_crash_model.h5')

# Create directory structure for SavedModel
model_dir = 'sagemaker_model/1'  # '1' is the version number
os.makedirs(model_dir, exist_ok=True)

# Save as SavedModel format (Keras 3 syntax)
tf.saved_model.save(model, model_dir)

print("✓ Model converted to SavedModel format")

# Create tar.gz for SageMaker
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('sagemaker_model', arcname='.')

print("✓ Model packaged: model.tar.gz")
print("File size:", os.path.getsize('model.tar.gz') / (1024*1024), "MB")

# Cleanup
shutil.rmtree('sagemaker_model')
