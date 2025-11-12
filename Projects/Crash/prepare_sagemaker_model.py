import tarfile
import os
import shutil

# Create model directory structure
os.makedirs('model', exist_ok=True)

# Copy your model
shutil.copy('models/car_crash_model.h5', 'model/car_crash_model.h5')

# Create model.tar.gz for SageMaker
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('model', arcname='.')

print("✓ Model packaged for SageMaker: model.tar.gz")
print("File size:", os.path.getsize('model.tar.gz') / (1024*1024), "MB")
