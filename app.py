# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests
import re
from thefuzz import process
import torch.nn.functional as F
import base64

# Page Configuration
st.set_page_config(
    page_title="Fruision",
    page_icon="🍓",
    layout="wide",
)

# Constants and Mappings
MODEL_PATH = "resnet50_trained_CV.pth"
CLASS_NAMES_PATH = "class_names.txt"
THEMEALDB_API_LOOKUP_URL = "https://www.themealdb.com/api/json/v1/1/lookup.php?i="
THEMEALDB_API_INGREDIENTS_URL = "https://www.themealdb.com/api/json/v1/1/list.php?i=list"
THEMEALDB_API_FILTER_URL = "https://www.themealdb.com/api/json/v1/1/filter.php?i="

# Load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

# Caching Functions
@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model(num_classes):
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# @st.cache_data
# def get_all_ingredients():
#     try:
#         response = requests.get(THEMEALDB_API_INGREDIENTS_URL)
#         response.raise_for_status()
#         data = response.json()
#         ingredients = [item['strIngredient'] for item in data['meals']]
#         ingredients.extend(["Apples", "Bananas", "Peaches", "Pears"])
#         return sorted(list(set(ingredients)))
#     except requests.RequestException:
#         return []

@st.cache_data
def get_all_ingredients():
    try:
        response = requests.get(THEMEALDB_API_INGREDIENTS_URL)
        response.raise_for_status()
        data = response.json()
        return sorted(
            item["strIngredient"].strip().lower()
            for item in data["meals"]
            if item["strIngredient"]
        )
    except requests.RequestException:
        return []

# Processing Functions
def transform_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return transform(image).unsqueeze(0)

# def get_best_api_ingredient(predicted_class, all_ingredients):
#     if not all_ingredients: return None
#     cleaned_prediction = re.sub(r'\s*\d+$', '', predicted_class).strip()
#     best_match, score = process.extractOne(cleaned_prediction, all_ingredients)
#     if score > 88: return best_match
#     general_term = predicted_class.split(' ')[0]
#     if general_term in all_ingredients: return general_term
#     if f"{general_term}s" in all_ingredients: return f"{general_term}s"
#     return None

def get_best_api_ingredient(predicted_class, api_ingredients):
    if not api_ingredients:
        return None

    pred = predicted_class.lower().strip()

    # Exact match first (best case)
    if pred in api_ingredients:
        return pred

    # Plural handling
    if f"{pred}s" in api_ingredients:
        return f"{pred}s"

    # Fuzzy match ONLY if close enough
    match, score = process.extractOne(pred, api_ingredients)
    if score >= 92:
        return match

    return None


def fetch_recipes(ingredient):
    try:
        response = requests.get(f"{THEMEALDB_API_FILTER_URL}{ingredient}")
        response.raise_for_status()
        data = response.json()
        return data['meals']
    except requests.RequestException:
        return None

@st.cache_data
def fetch_recipe_detail(meal_id):
    try:
        response = requests.get(f"{THEMEALDB_API_LOOKUP_URL}{meal_id}")
        response.raise_for_status()
        return response.json()["meals"][0]
    except requests.RequestException:
        return None

if "selected_meal_id" not in st.session_state:
    st.session_state.selected_meal_id = None


# UI Logic
local_css("style.css")

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("wallpaper.jpg")


st.markdown(f"""
<style>

/* ================= BACKGROUND ================= */
.stApp {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* ================= PAGE (SCROLL SAFE) ================= */
section[data-testid="stMain"] {{
    min-height: 100vh;
    padding: 4rem 3rem;
    overflow-y: auto !important;
}}

# /* Center horizontally ONLY */
# section[data-testid="stMain"] > div {{
#     display: flex;
#     justify-content: center;
# }}

/* ================= MAIN CONTAINER ================= */
div[data-testid="stVerticalBlock"] {{
    background: rgba(255, 255, 255, 0.96);
    padding: 5rem 5rem 6rem 5rem;
    border-radius: 32px;

    max-width: 1700px;
    width: 100%;

    margin: 6vh auto;   /* visual vertical centering */
    box-shadow: 0 24px 60px rgba(0,0,0,0.35);

    font-size: 1.6rem;
    text-align: left;
}}

/* ================= TYPOGRAPHY ================= */
h1 {{
    font-size: 4.2rem !important;
}}

h2 {{
    font-size: 3.2rem !important;
    margin-top: 3.5rem !important;
}}

h3 {{
    font-size: 2.4rem !important;
}}

p, label, span {{
    font-size: 1.65rem !important;
    line-height: 1.8;
}}

/* ================= FILE UPLOADER ================= */
/* ================= FILE UPLOADER (SAFE & CONTAINED) ================= */

div[data-testid="stFileUploader"] {{
    width: 100% !important;
}}

/* Dropzone container */
div[data-testid="stFileUploader"] section {{
    width: 100% !important;
    min-height: 220px !important;   /* 🔑 taller so button fits */
    padding: 2.4rem 2.6rem !important;

    border-radius: 20px !important;
    box-sizing: border-box;

    /* IMPORTANT: allow natural flow */
    display: block !important;
    overflow: hidden !important;
}}

/* Drag & drop text */
div[data-testid="stFileUploader"] section p {{
    font-size: 1.35rem !important;
    margin-bottom: 1.4rem !important;
    max-width: 100%;
}}

/* Browse files button */
div[data-testid="stFileUploader"] button {{
    font-size: 1.1rem !important;      /* 🔽 smaller than main UI */
    padding: 0.55rem 1.3rem !important;
    min-height: 44px !important;

    max-width: 100%;
    white-space: nowrap;

    display: inline-flex !important;
    align-items: center;
    justify-content: center;
}}

/* File size hint */
div[data-testid="stFileUploader"] small {{
    font-size: 1.1rem !important;
    margin-top: 0.9rem !important;
    display: block;
}}


/* ================= INPUTS & BUTTONS ================= */
input, textarea, select {{
    font-size: 1.55rem !important;
    padding: 1.1rem !important;
    min-height: 60px;
}}

button {{
    font-size: 1.6rem !important;
    padding: 1.1rem 2.3rem !important;
    min-height: 64px;
    border-radius: 16px !important;
}}

/* ================= IMAGES ================= */
img {{
    border-radius: 18px;
}}

/* ================= LINKS ================= */
a {{
    font-size: 1.45rem !important;
}}

/* ================= PROGRESS BAR ================= */
div[role="progressbar"] {{
    height: 26px !important;
}}

div[role="progressbar"] span {{
    font-size: 1.3rem !important;
}}

/* ================= RESPONSIVE ================= */
@media (max-width: 1024px) {{
    div[data-testid="stVerticalBlock"] {{
        padding: 4rem;
        margin: 4rem auto;
    }}

    h1 {{ font-size: 3.4rem !important; }}
    h2 {{ font-size: 2.7rem !important; }}
}}

@media (max-width: 768px) {{
    div[data-testid="stVerticalBlock"] {{
        padding: 3rem 2.4rem;
        border-radius: 26px;
    }}

    h1 {{ font-size: 2.8rem !important; }}
    h2 {{ font-size: 2.3rem !important; }}
    h3 {{ font-size: 1.9rem !important; }}

    p, label, span {{
        font-size: 1.4rem !important;
    }}
}}

</style>
""", unsafe_allow_html=True)




# ===== APP LOGIC =====
class_names = load_class_names()
model = load_model(len(class_names))
valid_api_ingredients = get_all_ingredients()

st.title("Welcome to Fruision!")
st.markdown("Upload an image of a fruit and we'll find recipes for it!")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to get started!")
    st.stop()

col1, col2 = st.columns([0.8, 1])

with col1:
    st.image(uploaded_file, caption="Your Upload", use_column_width=True)

with col2:
    with st.spinner("🧠 Analyzing image..."):
        image_tensor = transform_image(uploaded_file)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_name = class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100
        api_ingredient = get_best_api_ingredient(
            predicted_class_name,
            valid_api_ingredients
        )

    st.subheader("🔍 Analysis Result")
    st.markdown(f"**Model Prediction:** `{predicted_class_name}`")
    st.progress(confidence.item(), text=f"Confidence: {confidence_percent:.2f}%")

    if api_ingredient:
        st.success(f"**Found Ingredient:** `{api_ingredient}`")
    else:
        st.error(f"Could not find a matching ingredient for `{predicted_class_name}`.")

st.markdown(" ")

if api_ingredient:
    # st.header(f"🍳 Recipes with {api_ingredient}")
    st.markdown("<div class='recipes-spacer'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 class='recipes-title'>🍳 Recipes with {api_ingredient}</h2>",
        unsafe_allow_html=True
    )

    with st.spinner("Searching for recipes..."):
        recipes = fetch_recipes(api_ingredient)


    if recipes:
        max_recipes_to_show = 12

        for i in range(0, min(len(recipes), max_recipes_to_show), 4):
            cols = st.columns(4, gap="large")

            for j in range(4):
                if i + j < len(recipes):
                    recipe = recipes[i + j]

                    with cols[j]:
                        st.markdown("<div class='recipe-card'>", unsafe_allow_html=True)

                        st.image(recipe["strMealThumb"], use_column_width=True)

                        

                        # if st.button(
                        #     "View recipe",
                        #     key=f"meal_{recipe['idMeal']}"
                        # ):
                        #     st.session_state.selected_meal_id = recipe["idMeal"]

                        st.markdown(
                            f"<div class='recipe-title'>{recipe['strMeal']}</div>",
                            unsafe_allow_html=True
                        )

                        if st.button(
                            "View recipe",
                            key=f"meal_{recipe['idMeal']}"
                        ):
                            st.session_state.selected_meal_id = recipe["idMeal"]


                        st.markdown("</div>", unsafe_allow_html=True)


    # if st.session_state.selected_meal_id:
    #     st.markdown("---")
    # st.header("🍽️ Recipe Details")

    # detail = fetch_recipe_detail(st.session_state.selected_meal_id)

    # if detail:
    #     st.subheader(detail["strMeal"])
    #     st.image(detail["strMealThumb"], width=420)

    #     st.markdown("### 🧂 Ingredients")
    #     ingredients = []
    #     for i in range(1, 21):
    #         ing = detail.get(f"strIngredient{i}")
    #         meas = detail.get(f"strMeasure{i}")
    #         if ing and ing.strip():
    #             ingredients.append(f"- {meas} {ing}")

    #     st.markdown("\n".join(ingredients))

    #     st.markdown("### 📖 Instructions")
    #     st.write(detail["strInstructions"])



    if st.session_state.selected_meal_id:
        st.markdown(" ")
        with st.container():
            st.header("🍽️ Recipe Details")


        detail = fetch_recipe_detail(st.session_state.selected_meal_id)

        if detail:
            st.subheader(detail["strMeal"])
            st.image(detail["strMealThumb"], width=420)

            st.markdown("### 🧂 Ingredients")
            ingredients = []
            for i in range(1, 21):
                ing = detail.get(f"strIngredient{i}")
                meas = detail.get(f"strMeasure{i}")
                if ing and ing.strip():
                    ingredients.append(f"- {meas} {ing}")

            st.markdown("\n".join(ingredients))

            st.markdown("### 📖 Instructions")
            st.write(detail["strInstructions"])


    
    else:
        st.warning(
            f"No cooking recipes found for `{predicted_class_name}`.\n"
            "This ingredient may not be commonly used as a main dish."
        )
