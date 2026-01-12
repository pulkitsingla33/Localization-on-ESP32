
"""
train_mlp_localization.py

Train a small MLP for WiFi RSSI fingerprint localization and export a fully-quantized
TFLite model for deployment on microcontrollers (TFLite Micro).

Assumptions:
 - CSV has first column named "location" (label). Remaining columns are RSSI features in fixed order.
 - Missing APs are filled with a sentinel (we use -105 here).
 - Feature order must match the order you use on-device.

Outputs:
 - mlp_keras.h5               (optional full Keras model)
 - mlp_model.tflite           (INT8 quantized tflite model)
 - mlp_features.txt           (feature names, one per line)
 - mlp_label_map.txt          (index,label CSV)
"""

import os
import random
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

# -----------------------------------------------------------------------------
# Config (tweak as needed)
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

CSV_FILE = "fingerprint_pass_matrix_fixed.csv"   # change if needed
LABEL_COL = "location"
MISSING_RSSI = -105.0



# MLP hyperparams
INPUT_DROPOUT = 0.0
HIDDEN_UNITS = [64, 32]   # two dense layers; tune smaller/larger
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 8
LEARNING_RATE = 1e-3

# Train/test split strategy:
# If your rows include timestamped passes and suffixes like loc_00001 etc,
# we will group by "base location" (label without final _000xxx) for GroupSplit.
USE_GROUP_SPLIT = True         # recommended: prevents leakage
GROUP_SPLIT_TEST_SIZE = 0.2

# Representative dataset params for quantization
REPRESENTATIVE_SAMPLE_SIZE = 500  # number of samples used by converter for INT8 calibration

# Output files
OUT_KERAS = "mlp_keras.h5"
OUT_TFLITE = "mlp_model.tflite"
OUT_FEATURES = "mlp_features.txt"
OUT_LABELMAP = "mlp_label_map.txt"
OUT_META = "mlp_meta.json"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def base_location(label):
    # remove trailing _000123 style suffix if present, else return as-is
    if isinstance(label, str) and "_" in label:
        parts = label.rsplit("_", 1)
        if len(parts[-1]) == 6 and parts[-1].isdigit():
            return parts[0]
    return label

def load_data(csv_file):
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
    assert LABEL_COL in df.columns, f"Label column '{LABEL_COL}' not found!"
    
    # feature columns are everything except label
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    # ensure order preserved
    X = df[feat_cols].copy()
    y = df[LABEL_COL].copy()
    return X, y, feat_cols

def preprocess_X(X_df):
    # Fill missing (NaN) with MISSING_RSSI (float)
    X = X_df.fillna(MISSING_RSSI).astype(float).values
    # Clip extreme RSSI (optional) - we avoid hard clipping; ensure numeric stability
    # Scale RSSI into roughly [-1, 1]. RSSIs are negative: e.g. -30..-105
    # We'll map rssi_scaled = (rssi - RF_MIN) / (RF_MAX - RF_MIN) scaled to [-1,1]
    rf_min = -110.0
    rf_max = -30.0
    X_clipped = np.clip(X, rf_min, rf_max)
    X_scaled = 2.0 * (X_clipped - rf_min) / (rf_max - rf_min) - 1.0
    return X_scaled.astype(np.float32), rf_min, rf_max

# -----------------------------------------------------------------------------
# Model builder
# -----------------------------------------------------------------------------
def build_mlp(input_dim, n_classes, hidden_units=HIDDEN_UNITS, dropout_rate=DROPOUT_RATE):
    inputs = tf.keras.Input(shape=(input_dim,), name="rssi_input")
    x = inputs
    if INPUT_DROPOUT and INPUT_DROPOUT > 0:
        x = tf.keras.layers.Dropout(INPUT_DROPOUT)(x)
    for i, u in enumerate(hidden_units):
        x = tf.keras.layers.Dense(u, activation="relu", name=f"dense_{i}")(x)
        if dropout_rate and dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# -----------------------------------------------------------------------------
# Representative dataset generator for INT8 quantization
# -----------------------------------------------------------------------------
def representative_generator(X_train):
    # yields batches of shape (1, input_dim) as float32
    def gen():
        count = 0
        n = X_train.shape[0]
        idxs = np.random.RandomState(SEED).permutation(n)
        for i in idxs[:REPRESENTATIVE_SAMPLE_SIZE]:
            sample = X_train[i:i+1]
            yield [sample.astype(np.float32)]
    return gen

# -----------------------------------------------------------------------------
# TFLite evaluation helper (to test quantized model)
# -----------------------------------------------------------------------------
def evaluate_tflite(tflite_path, X_test, y_test):
    # run TFLite interpreter (works for both float and int8) — expects model_input scaling done externally
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine whether input is quantized
    input_scale, input_zero_point = None, None
    if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
    preds = []
    for i in range(X_test.shape[0]):
        x = X_test[i:i+1]
        if input_scale is not None:
            # quantize
            x_q = (x / input_scale + input_zero_point).astype(np.int8)
            interpreter.set_tensor(input_details[0]['index'], x_q)
        else:
            interpreter.set_tensor(input_details[0]['index'], x.astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        pred = int(np.argmax(out, axis=1)[0])
        preds.append(pred)
    acc = accuracy_score(y_test, preds)
    return acc, preds

# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------
def main():
    print("Loading data:", CSV_FILE)
    X_df, y_ser, feat_cols = load_data(CSV_FILE)
    print("Features:", len(feat_cols), "labels:", len(y_ser.unique()))

    # per-location shuffle (prevent block ordering problems)
    df_all = X_df.copy()
    df_all[LABEL_COL] = y_ser.values
    shuffled_chunks = []
    for loc, group in df_all.groupby(LABEL_COL):
        shuffled_chunks.append(group.sample(frac=1.0, random_state=SEED))
    df_all = pd.concat(shuffled_chunks).reset_index(drop=True)

    # grouped train/test split: group by 'base location' so different passes don't leak
    # Force normal stratified split (recommended with your cleaned labels)
    df_train, df_test = train_test_split(
        df_all,
        test_size=0.2,
        random_state=SEED,
        stratify=df_all[LABEL_COL],
    )


    X_train_df = df_train[feat_cols]
    y_train_ser = df_train[LABEL_COL]
    X_test_df = df_test[feat_cols]
    y_test_ser = df_test[LABEL_COL]

    # encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_ser)
    y_test_enc = le.transform(y_test_ser)

    label_map = {i: cls for i, cls in enumerate(le.classes_)}
    print("Label mapping:", label_map)

    # preprocess X (fill missing, scale)
    X_train_scaled, rf_min, rf_max = preprocess_X(X_train_df)
    X_test_scaled, _, _ = preprocess_X(X_test_df)

    # build model
    input_dim = X_train_scaled.shape[1]
    n_classes = len(le.classes_)
    print("Building MLP: input_dim=", input_dim, "n_classes=", n_classes)
    model = build_mlp(input_dim, n_classes)
    model.summary()

    # callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
    ]

    # fit
    history = model.fit(
        X_train_scaled, y_train_enc,
        validation_split=0.15,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    # evaluate float model
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Float model test accuracy: {acc:.4f}")
    print("Classification report (float model):")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_enc, y_pred))

    # save keras model for backup
    model.save(OUT_KERAS)
    print("Saved Keras model:", OUT_KERAS)

    # Prepare representative dataset function
    rep_gen = representative_generator(X_train_scaled)

    # Convert to TFLite INT8 (full integer quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    # target ops - allow full integer
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # enforce int8 input and output
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(OUT_TFLITE, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model:", OUT_TFLITE)

    # Evaluate quantized model on test data
    q_acc, q_preds = evaluate_tflite(OUT_TFLITE, X_test_scaled, y_test_enc)
    print(f"Quantized model (TFLite INT8) test accuracy: {q_acc:.4f}")

    # Save feature and label lists and meta
    with open(OUT_FEATURES, "w") as f:
        for mac in feat_cols:
            f.write(mac + "\n")
    with open(OUT_LABELMAP, "w") as f:
        for idx, cls in label_map.items():
            f.write(f"{idx},{cls}\n")
    meta = {
        "rf_min": rf_min,
        "rf_max": rf_max,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "hidden_units": HIDDEN_UNITS,
        "dropout_rate": DROPOUT_RATE
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved feature & label map & meta files.")
    print("Done.")

if __name__ == "__main__":
    main()
