import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from recipes import recipes
import os
from dotenv import load_dotenv
import base64
import openai
from openai.error import AuthenticationError

# --- Setup ---
st.set_page_config(page_title="MoodMeal – AI-Powered Recipe & Mood Recommender 🍽", page_icon="🥗", layout="wide", initial_sidebar_state="collapsed")
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Custom DepthwiseConv2D for model loading ---
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return cls(**config)

@st.cache_resource
def load_custom_model():
    return tf.keras.models.load_model(
        'keras_model.h5',
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
    )

def load_labels(path='labels.txt'):
    with open(path, 'r') as file:
        return [line.strip().split(' ', 1)[1] for line in file.readlines()]

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model, image, labels):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    return labels[class_idx], prediction[class_idx]

def get_food_options(mood_key):
    return list(recipes.get(mood_key, {}).keys())

# --- Load Model and Labels ---
labels = load_labels('labels.txt')
model = load_custom_model()

# --- Logo ---
try:
    with open('logo.jpg', 'rb') as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    logo_src = f"data:image/jpeg;base64,{logo_base64}"
except FileNotFoundError:
    logo_src = "https://placehold.co/120x50/aabbcc/ffffff?text=Logo+Missing"

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(f"""
    <div class='header'>
        <img src="{logo_src}" class="logo">
        <h1>MoodMeal 🍽 — AI-Powered Recipe Recommender</h1>
    </div>
    <hr>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content-animated" style="padding-top: 4px;">', unsafe_allow_html=True)
st.subheader("🎯 Choose an option:")

option = st.radio("Choose an Option:", [
    "📷 Detect Mood Using Webcam",
    "🖼 Upload an Image",
    "🌐 Choose Manually",
    "🤖 AI-Based Mood Recipe Suggestion"
], key="main_option_radio")

# --- Option 1: Webcam ---
if option == "📷 Detect Mood Using Webcam":
    st.markdown("---")
    st.subheader("Capture your mood with your webcam!")
    camera_img = st.camera_input("Take a photo", key="webcam_input")
    if camera_img:
        st.image(camera_img, caption="Captured Image", use_column_width=True)
        image = Image.open(camera_img).convert('RGB')
        img_np = np.array(image)
        mood, confidence = predict_image(model, img_np, labels)
        st.success(f"Detected Mood: {mood} ({confidence*100:.2f}%)")

        food_options = get_food_options(mood)
        if food_options:
            food_choice = st.selectbox("Choose Food Type 🍽:", food_options, key="webcam_food_select")
            if st.button("Show Recipes", key="webcam_show_recipes_btn"):
                recommended = recipes[mood].get(food_choice, [])
                if recommended:
                    st.subheader("🍽 Recommended Recipes:")
                    for recipe in recommended:
                        st.markdown(f"**{recipe['name']}**\n\n{recipe['instructions']}")
                else:
                    st.warning("No recipes found.")

# --- Option 2: Upload Image ---
elif option == "🖼 Upload an Image":
    st.markdown("---")
    st.subheader("Upload an image to detect your mood!")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_np = np.array(image)
        mood, confidence = predict_image(model, img_np, labels)
        st.success(f"Detected Mood: {mood} ({confidence*100:.2f}%)")

        food_options = get_food_options(mood)
        if food_options:
            food_choice = st.selectbox("Choose Food Type 🍽:", food_options, key="upload_food_select")
            if st.button("Show Recipes", key="upload_show_recipes_btn"):
                recommended = recipes[mood].get(food_choice, [])
                if recommended:
                    st.subheader("🍽 Recommended Recipes:")
                    for recipe in recommended:
                        st.markdown(f"**{recipe['name']}**\n\n{recipe['instructions']}")
                else:
                    st.warning("No recipes found.")

# --- Option 3: Choose Manually ---
elif option == "🌐 Choose Manually":
    st.markdown("---")
    st.subheader("Select your mood and get recipe suggestions!")
    mood = st.selectbox("Pick a Mood:", list(recipes.keys()), key="manual_mood_select")
    food_options = get_food_options(mood)
    if food_options:
        food_choice = st.selectbox("Select Food Type:", food_options, key="manual_food_select")
        if st.button("Show Recipes", key="manual_show_recipes_btn"):
            recommended = recipes[mood].get(food_choice, [])
            if recommended:
                st.subheader(f"🍽 Recipes for {mood} Mood and {food_choice} Food:")
                for recipe in recommended:
                    st.markdown(f"**{recipe['name']}**\n\n{recipe['instructions']}")
            else:
                st.warning("No recipes found.")

# --- Option 4: AI Mood Recipe ---
elif option == "🤖 AI-Based Mood Recipe Suggestion":
    st.markdown("---")
    st.subheader("Tell our AI how you're feeling for a personalized recipe!")
    mood_input = st.text_input("How are you feeling?", key="ai_mood_input")
    if st.button("Get AI Recipe Suggestion", key="get_ai_recipe_btn"):
        if not mood_input or len(mood_input.strip()) < 3:
            st.warning("Please enter a valid mood.")
        else:
            with st.spinner("Thinking of a recipe for you..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": (
                                "You are a recipe assistant. Given any sentence, extract the emotion or mood being expressed "
                                "(like happy, sad, stressed, etc.). If a mood is identified, suggest a detailed recipe suitable for that mood. "
                                "Structure the response as follows:\n\n"
                                "Recipe Name: <name>\n\n"
                                "Ingredients:\n- item 1\n- item 2\n...\n\n"
                                "Instructions:\n1. Step one\n2. Step two\n...\n\n"
                                "If the input contains no recognizable emotion or mood, respond with: 'Can't suggest a recipe for that input.'"
                            )},
                            {"role": "user", "content": f"Suggest a recipe for someone feeling: {mood_input}"}
                        ],
                        max_tokens=300,
                        temperature=0.7
                    )
                    suggestion = response['choices'][0]['message']['content'].strip()
                    if "Can't suggest" in suggestion:
                        st.info(suggestion)
                    else:
                        st.success("Here's a recipe for you:")
                        st.markdown(suggestion)
                except AuthenticationError:
                    st.error("OpenAI API Key is invalid or missing.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

st.markdown('</div>', unsafe_allow_html=True)
