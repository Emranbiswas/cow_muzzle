import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Path to the dataset directory (adjust as needed)
#dataset_dir = r'D:\cow_detection\cow_data'  # Update to your dataset path

# Function to extract class names dynamically from the dataset
def get_class_names(dataset_dir):
    class_names = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    class_names.sort()  # Optional: Sort class names alphabetically
    return class_names

# Get class names
#class_names = get_class_names(dataset_dir)

# Load the trained model
model = load_model('muzzle_model.h5')  # Path to your model file

# Function to preprocess image
def preprocess_image(image, target_size=(128, 128)):
    image = image.convert('RGB')  # Ensure 3 channels
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)
    return image

# App title
st.title("Cow Detection App")
st.write("Upload an image of a cow, and the app will predict the cow's name!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width =True)
    #st.write("Processing...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions, axis=-1)[0]
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    # Show prediction
    st.write(f"**Predicted Cow Name:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
