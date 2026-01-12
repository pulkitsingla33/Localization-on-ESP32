import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load windowed data
X = np.load("X_windows_no_gyro.npy")   # (N,50,6)
y = np.load("y_windows_no_gyro.npy")

# Flatten for MLP
X_flat = X.reshape(len(X), -1)      # → (N, 300)

#Normalize Data using Standard Scaler
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
# )

X_train = []
y_train = []
X_test = []
y_test = []

unique_classes = np.unique(y_enc)
for cls in unique_classes:
    # Get all indices for this class
    idxs = np.where(y_enc == cls)[0]
    
    # Split point (80%)
    split = int(len(idxs) * 0.8)
    
    # Train on first part, Test on second part
    X_train.extend(X_flat[idxs[:split]])
    y_train.extend(y_enc[idxs[:split]])
    
    X_test.extend(X_flat[idxs[split:]])
    y_test.extend(y_enc[idxs[split:]])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape:  {X_test.shape}")

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(50,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(unique_classes), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.1)

# Test accuracy
pred = model.predict(X_test)
pred_classes = pred.argmax(axis=1)

print("Accuracy =", accuracy_score(y_test, pred_classes))
print(classification_report(y_test, pred_classes))

# Save model and label map
model.save("activity_mlp.h5")
np.save("activity_labels.npy", le.classes_)


np.save("scaler_mean_no_gyro.npy", scaler.mean_)
np.save("scaler_std_no_gyro.npy", scaler.scale_)
