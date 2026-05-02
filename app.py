import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('efficientnet.keras')
    image = Image.open(test_image).convert('RGB')
    image = image.resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = prediction[0][result_index]
    return result_index, confidence

# Class Names
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Cotton___Bacterial_blight', 'Cotton___Curl_virus', 'Cotton___Fusarium_wilt', 'Cotton___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Rice___Bacterial_leaf_blight', 'Rice___Brown_spot', 'Rice___Leaf_blast', 'Rice___Leaf_scald',
    'Rice___Sheath_blight', 'Rice___healthy',
    'Sugarcane___Mosaic', 'Sugarcane___Red_rot', 'Sugarcane___Rust', 'Sugarcane___Yellow', 'Sugarcane___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Page Title
st.title("🌿 Plant Disease Detection")
st.markdown("""
Upload an image of a plant leaf, and our AI model will detect if it is healthy or diseased, along with the confidence score.
""")

# File Uploader
test_image = st.file_uploader("📂 Upload a Plant Leaf Image", type=["jpg", "jpeg", "png"])
confidence_level = 50
if test_image is not None:
    st.image(test_image, caption='Uploaded Image', use_column_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing Image..."):
            result_index, confidence = model_prediction(test_image)
            confidence_percent = confidence * 100

            if confidence_percent < confidence_level:
                st.warning("⚠️ The image does not contain a clear leaf or is not suitable for analysis. Please upload a clearer image.")
            else:
                st.success(f"Prediction: **{class_name[result_index]}**")
                st.info(f"Confidence: **{confidence_percent:.2f}%**")
