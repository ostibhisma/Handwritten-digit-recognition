import os
import cv2
import joblib
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Enter the name of image you want to test")
args = vars(ap.parse_args())

image_name = args["image"]

# Load the model from the file
svm_from_joblib = joblib.load(os.path.join("Models","svm_model.pkl"))

image = cv2.imread(os.path.join("handwritten_digit_corpus", image_name))
# resizing the image as height 224 and width 224
image_resized = cv2.resize(image,(28,28))

img = np.array(image_resized)/255.0

nx, ny, _ = img.shape
img = img.reshape((-1, nx*ny*_))

 
# Use the loaded model to make predictions
result = svm_from_joblib.predict(img)
print(f"Your Image is {result[0]}")