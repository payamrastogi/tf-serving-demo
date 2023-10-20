import os
import json
import time
import shutil
import requests
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.applications.MobileNet()


def preprocess(image, mean=0.5, std=0.5, shape=(224, 224)):
    """Scale, normalize and resizes images."""
    image = image / 255.0  # Scale
    image = (image - mean) / std  # Normalize
    image = tf.image.resize(image, shape)  # Resize
    return image


# Download human-readable labels for ImageNet.
imagenet_labels_url = ("https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
response = requests.get(imagenet_labels_url)
# Skiping backgroung class
labels = [x for x in response.text.split("\n") if x != ""][1:]
# Convert the labels to the TensorFlow data format
tf_labels = tf.constant(labels, dtype=tf.string)


def postprocess(prediction, labels=tf_labels):
    """Convert from probs to labels."""
    indices = tf.argmax(prediction, axis=-1)  # Index with highest prediction
    label = tf.gather(params=labels, indices=indices)  # Class name
    return label


response = requests.get("https://i.imgur.com/j9xCCzn.jpeg", stream=True)

with open("banana.jpeg", "wb") as f:
    shutil.copyfileobj(response.raw, f)

sample_img = plt.imread("./banana.jpeg")
print(f"Original image shape: {sample_img.shape}")
print(f"Original image pixel range: ({sample_img.min()}, {sample_img.max()})")
plt.imshow(sample_img)
plt.show()

preprocess_img = preprocess(sample_img)
print(f"Preprocessed image shape: {preprocess_img.shape}")
print(
    f"Preprocessed image pixel range: ({preprocess_img.numpy().min()},",
    f"{preprocess_img.numpy().max()})",
)

batched_img = tf.expand_dims(preprocess_img, axis=0)
batched_img = tf.cast(batched_img, tf.float32)
print(f"Batched image shape: {batched_img.shape}")

model_outputs = model(batched_img)
print(f"Model output shape: {model_outputs.shape}")
print(f"Predicted class: {postprocess(model_outputs)}")


model_dir = "./model"
model_version = 1
model_export_path = f"{model_dir}/{model_version}"

tf.saved_model.save( model, export_dir=model_export_path,)

print(f"SavedModel files: {os.listdir(model_export_path)}")
