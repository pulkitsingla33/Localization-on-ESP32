import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("activity_mlp.h5")
X_train = np.load("X_windows_no_gyro.npy").reshape(-1, 50).astype(np.float32)

def rep():
    for i in range(500):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("activity_mlp_no_gyro.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved activity_mlp.tflite")
