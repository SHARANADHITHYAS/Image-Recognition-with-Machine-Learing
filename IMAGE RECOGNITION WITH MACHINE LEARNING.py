# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess the image
img_path = 'image.jpg'  # Replace 'image.jpg' with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
predictions = model.predict(x)
decoded_predictions = decode_predictions(predictions, top=3)[0]

# Print the top 3 predicted classes
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f'{i + 1}: {label} ({score:.2f})')


#This code uses the MobileNetV2 model pre-trained on the ImageNet dataset, which is capable of recognizing a wide range of objects. Make sure to replace 'image.jpg' with the path to your image file.