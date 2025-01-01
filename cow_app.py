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
class_names = 

['cattle_5009',
 'cattle_5026',
 'cattle_5028',
 'cattle_5066',
 'cattle_5073',
 'cattle_5077',
 'cattle_5083',
 'cattle_5090',
 'cattle_5097',
 'cattle_5100',
 'cattle_5112',
 'cattle_5132',
 'cattle_5133',
 'cattle_5138',
 'cattle_5143',
 'cattle_5153',
 'cattle_5164',
 'cattle_5165',
 'cattle_5170',
 'cattle_5171',
 'cattle_5197',
 'cattle_5207',
 'cattle_5208',
 'cattle_5215',
 'cattle_5224',
 'cattle_5234',
 'cattle_5235',
 'cattle_5249',
 'cattle_5273',
 'cattle_5275',
 'cattle_5282',
 'cattle_5283',
 'cattle_5297',
 'cattle_5298',
 'cattle_5307',
 'cattle_5314',
 'cattle_5325',
 'cattle_5355',
 'cattle_5359',
 'cattle_5360',
 'cattle_5362',
 'cattle_5373',
 'cattle_5374',
 'cattle_5403',
 'cattle_5404',
 'cattle_5407',
 'cattle_5408',
 'cattle_5410',
 'cattle_5411',
 'cattle_5425',
 'cattle_5427',
 'cattle_5432',
 'cattle_5477',
 'cattle_5507',
 'cattle_5508',
 'cattle_5509',
 'cattle_5519',
 'cattle_5529',
 'cattle_5537',
 'cattle_5556',
 'cattle_5559',
 'cattle_5581',
 'cattle_5604',
 'cattle_5605',
 'cattle_5620',
 'cattle_5630',
 'cattle_5633',
 'cattle_5634',
 'cattle_5639',
 'cattle_5654',
 'cattle_5658',
 'cattle_5670',
 'cattle_5677',
 'cattle_5695',
 'cattle_5697',
 'cattle_5717',
 'cattle_5745',
 'cattle_5761',
 'cattle_5762',
 'cattle_5774',
 'cattle_5777',
 'cattle_5781',
 'cattle_5784',
 'cattle_5803',
 'cattle_5804',
 'cattle_5806',
 'cattle_5809',
 'cattle_5815',
 'cattle_5816',
 'cattle_5836',
 'cattle_5844',
 'cattle_5886',
 'cattle_5925',
 'cattle_5932',
 'cattle_5953',
 'cattle_5971',
 'cattle_5986',
 'cattle_6011',
 'cattle_6012',
 'cattle_6017',
 'cattle_6022',
 'cattle_6038',
 'cattle_6066',
 'cattle_6071',
 'cattle_6084',
 'cattle_6098',
 'cattle_6124',
 'cattle_6161',
 'cattle_6167',
 'cattle_6171',
 'cattle_6184',
 'cattle_6189',
 'cattle_6191',
 'cattle_6196',
 'cattle_6197',
 'cattle_6199',
 'cattle_6210',
 'cattle_6213',
 'cattle_6216',
 'cattle_6220',
 'cattle_6226',
 'cattle_6237',
 'cattle_6253',
 'cattle_6266',
 'cattle_6276',
 'cattle_6277',
 'cattle_6278',
 'cattle_6282',
 'cattle_6283',
 'cattle_6287',
 'cattle_6294',
 'cattle_6295',
 'cattle_6313',
 'cattle_6331',
 'cattle_6333']

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
