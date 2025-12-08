import os
import pickle
import io
import requests
import hashlib

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
# Replace this with your actual CloudFront/CDN URL after setting it up in Phase 1
CDN_URL = "https://d234a7buedt55s.cloudfront.net/model.p"
LOCAL_MODEL_PATH = './model.p'
USE_CDN = False  # Set to False to run completely locally

# prepare data
test_images_dir = './test_images'  # Changed to dir to match loop logic, or keep as file
input_dir = './outfits/'
categories = ['trendy', 'not_trendy']

# 1. DATA PREPARATION & TRAINING
# (Only necessary if you are regenerating the model)
if not os.path.exists(LOCAL_MODEL_PATH) and not USE_CDN:
    print("Training model locally...")
    data = []
    labels = []

    # Check if input directory exists to prevent crash
    if os.path.exists(input_dir):
        for category_idx, category in enumerate(categories):
            category_path = os.path.join(input_dir, category)
            if not os.path.exists(category_path): continue

            for file in os.listdir(category_path):
                if file.startswith('.'):
                    continue
                try:
                    img_path = os.path.join(category_path, file)
                    img = imread(img_path)
                    img = resize(img, (15, 15))
                    data.append(img.flatten())
                    labels.append(category_idx)
                except Exception as e:
                    print(f"Couldn't read image {img_path}: {e}")

        if len(data) > 0:
            data = np.asarray(data)
            labels = np.asarray(labels)

            # train / test split
            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True,
                                                                stratify=labels)

            # train classifier
            classifier = SVC()
            parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
            grid_search = GridSearchCV(classifier, parameters)
            grid_search.fit(x_train, y_train)

            # test performance
            best_estimator = grid_search.best_estimator_
            y_prediction = best_estimator.predict(x_test)
            score = accuracy_score(y_prediction, y_test)
            print('{}% of samples were correctly classified'.format(str(score * 100)))

            # Save model locally so it can be uploaded to CDN later
            pickle.dump(best_estimator, open(LOCAL_MODEL_PATH, 'wb'))
            print(f"Model saved to {LOCAL_MODEL_PATH}. Please upload this to your CDN Origin.")
        else:
            print("No training data found.")
    else:
        print("Training directory not found.")

# 2. LOAD MODEL (CDN IMPLEMENTATION)
loaded_model = None

if USE_CDN:
    print(f"Attempting to download model from CDN: {CDN_URL}")
    try:
        response = requests.get(CDN_URL)
        if response.status_code == 200:
            # io.BytesIO allows us to read the downloaded bytes as if it were a file
            loaded_model = pickle.loads(response.content)
            print("Model loaded successfully from CDN.")
        else:
            print(f"CDN Error: Status code {response.status_code}")
    except Exception as e:
        print(f"Failed to fetch from CDN: {e}")

# Fallback to local if CDN failed or disabled
if loaded_model is None:
    print("Loading model from local storage...")
    try:
        with open(LOCAL_MODEL_PATH, 'rb') as f:
            loaded_model = pickle.load(f)
        print("Local model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found locally or on CDN.")
        exit()
# Hashing to validate authenticity between CDN and local model
if USE_CDN:
    response = requests.get(CDN_URL)
    model_data = response.content
    print(f"CDN Model Checksum: {hashlib.md5(model_data).hexdigest()}")
    loaded_model = pickle.loads(model_data)
else:
    with open(LOCAL_MODEL_PATH, 'rb') as f:
        model_data = f.read()
    print(f"Local Model Checksum: {hashlib.md5(model_data).hexdigest()}")
    loaded_model = pickle.loads(model_data)

# 3. PREDICTION
# Process all images in the test_images directory
if os.path.exists(test_images_dir):
    for file in os.listdir(test_images_dir):
        if file.startswith('.'): continue

        try:
            img_path = os.path.join(test_images_dir, file)
            img = imread(img_path)
            img = resize(img, (15, 15))
            flattened = img.flatten().reshape(1, -1)

            prediction = loaded_model.predict(flattened)
            prediction_category = categories[prediction[0]]
            print(f"Image ({file}) is classified as: **{prediction_category}**")

        except Exception as e:
            print(f"Error processing {file}: {e}")
else:
    print(f"Test directory '{test_images_dir}' does not exist.")