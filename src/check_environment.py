
import tensorflow as tf
import cv2
import numpy as np

print("DIT5411 Machine Learning")
print("=" * 35)

# Display library versions
print("TensorFlow:", tf.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

# Create a 100 x 100 black image
image = np.zeros((100, 100), dtype=np.uint8)

# Draw a white circle
cv2.circle(image, (50, 50), 30, 255, -1)

print("Image shape:", image.shape)

# Test TensorFlow
tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print("TensorFlow tensor:")
print(tensor.numpy())

print("=" * 35)
print("Environment Test Passed!")
