import requests
import json
import tensorflow as tf
import numpy as np
import shutil
import matplotlib.pyplot as plt


def preprocess(image, mean=0.5, std=0.5, shape=(224, 224)):
    """Scale, normalize and resizes images."""
    image = image / 255.0  # Scale
    image = (image - mean) / std  # Normalize
    image = tf.image.resize(image, shape)  # Resize
    return image


# Download human-readable labels for ImageNet.
imagenet_labels_url = ("https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
response = requests.get(imagenet_labels_url)
# Skipping background class
labels = [x for x in response.text.split("\n") if x != ""][1:]
# Convert the labels to the TensorFlow data format
tf_labels = tf.constant(labels, dtype=tf.string)


def postprocess(prediction, labels=tf_labels):
    """Convert from probs to labels."""
    indices = tf.argmax(prediction, axis=-1)  # Index with the highest prediction
    label = tf.gather(params=labels, indices=indices)  # Class name
    return label


def predict_rest(json_data, url):
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    rest_outputs = np.array(response["predictions"])
    return rest_outputs


response = requests.get("https://i.imgur.com/j9xCCzn.jpeg", stream=True)

with open("banana.jpeg", "wb") as f:
    shutil.copyfileobj(response.raw, f)

sample_img = plt.imread("./banana.jpeg")
print(f"Original image shape: {sample_img.shape}")
print(f"Original image pixel range: ({sample_img.min()}, {sample_img.max()})")
preprocess_img = preprocess(sample_img)
print(f"Preprocessed image shape: {preprocess_img.shape}")
print(
    f"Preprocessed image pixel range: ({preprocess_img.numpy().min()},",
    f"{preprocess_img.numpy().max()})",
)

batched_img = tf.expand_dims(preprocess_img, axis=0)
batched_img = tf.cast(batched_img, tf.float32)
print(f"Batched image shape: {batched_img.shape}")
data = json.dumps({"signature_name": "serving_default", "instances": batched_img.numpy().tolist()})
url = "http://localhost:8501/v1/models/model:predict"
rest_outputs = predict_rest(data, url)

print(f"REST output shape: {rest_outputs.shape}")
print(f"Predicted class: {postprocess(rest_outputs)}")
