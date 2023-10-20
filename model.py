import tensorflow as tf
import os

model = tf.keras.applications.MobileNet()

model_dir = "./model"
model_version = 1
model_export_path = f"{model_dir}/{model_version}"

tf.saved_model.save(model, export_dir=model_export_path,)

print(f"SavedModel files: {os.listdir(model_export_path)}")
