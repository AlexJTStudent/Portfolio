import numpy as np
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

def extract_simple_features(image_path):
    """Extract simple features from image without TensorFlow"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))  # Smaller for speed
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Extract features
    features = []
    
    # 1. Color histogram (RGB)
    for channel in range(3):
        hist, _ = np.histogram(img_array[:,:,channel], bins=8, range=(0, 256))
        features.extend(hist / hist.sum())  # Normalize
    
    # 2. Mean and std of each channel
    for channel in range(3):
        features.append(img_array[:,:,channel].mean())
        features.append(img_array[:,:,channel].std())
    
    # 3. Edge detection (simple gradient)
    gray = np.mean(img_array, axis=2)
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    features.extend([grad_x, grad_y])
    
    # 4. Texture (variance in patches)
    patch_size = 16
    for i in range(0, 64, patch_size):
        for j in range(0, 64, patch_size):
            patch = gray[i:i+patch_size, j:j+patch_size]
            features.append(patch.var())
    
    return np.array(features)

print("Extracting simple features from images...")
X = []
y = []

# Process class 1 (not crashed)
class1_dir = 'C:\\Users\\Alex\\Documents\\GitHub\\Portfolio\\Projects\\Crash\\Data\\AccidentDetection\\1'
count = 0
for img_name in os.listdir(class1_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(class1_dir, img_name)
        try:
            features = extract_simple_features(img_path)
            X.append(features)
            y.append(0)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} not-crashed images...")
        except Exception as e:
            print(f"Error: {e}")

print(f"Total not-crashed: {count}")

# Process class 2 (crashed)
class2_dir = 'C:\\Users\\Alex\\Documents\\GitHub\\Portfolio\\Projects\\Crash\\Data\\AccidentDetection\\2'
count = 0
for img_name in os.listdir(class2_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(class2_dir, img_name)
        try:
            features = extract_simple_features(img_path)
            X.append(features)
            y.append(1)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} crashed images...")
        except Exception as e:
            print(f"Error: {e}")

print(f"Total crashed: {count}")

X = np.array(X)
y = np.array(y)

print(f"\nTotal samples: {len(X)}")
print(f"Feature vector size: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

print("\nEvaluating model...")
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
with open('models/simple_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print(f"\n✓ Model saved to: models/simple_rf_model.pkl")
print(f"Model size: {os.path.getsize('models/simple_rf_model.pkl') / (1024*1024):.2f} MB")
