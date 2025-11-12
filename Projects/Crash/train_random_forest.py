import tensorflow as tf
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

print("Loading MobileNetV2 for feature extraction...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'  # This gives us a 1280-length vector
)

def extract_features(image_path):
    """Extract features from a single image"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()

print("Extracting features from images...")
X = []
y = []

# Process class 1 (not crashed)
class1_dir = 'C:\\Users\\Alex\\Documents\\GitHub\\Portfolio\\Projects\\Crash\\Data\\AccidentDetection\\1'
for img_name in os.listdir(class1_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(class1_dir, img_name)
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(0)  # 0 = not crashed
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"Processed {len([i for i in y if i == 0])} not-crashed images")

# Process class 2 (crashed)
class2_dir = 'C:\\Users\\Alex\\Documents\\GitHub\\Portfolio\\Projects\\Crash\\Data\\AccidentDetection\\2'
for img_name in os.listdir(class2_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(class2_dir, img_name)
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(1)  # 1 = crashed
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"Processed {len([i for i in y if i == 1])} crashed images")

X = np.array(X)
y = np.array(y)

print(f"\nTotal samples: {len(X)}")
print(f"Feature vector size: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
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
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print(f"\n✓ Model saved to: models/random_forest_model.pkl")
print(f"Model size: {os.path.getsize('models/random_forest_model.pkl') / (1024*1024):.2f} MB")
