import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
import os
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# ========== PARAMETERS ==========
FACE_CLASS = 11
CLOTHES_CLASSES = [4, 7]  # Upper-clothes and Dress

# ========== FUNCTIONS ==========
def extract_dominant_color(img, mask, num_colors=1):
    img_array = np.array(img)
    mask_array = np.array(mask)
    mask_binary = (mask_array > 0).astype(np.uint8)

    segmented_pixels = img_array[mask_binary == 1]
    if segmented_pixels.size == 0:
        return [[0, 0, 0]] * num_colors

    data = np.float32(segmented_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, flags)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)

    dominant_colors = []
    for i in range(num_colors):
        dominant_index = sorted_clusters[i][0]
        dominant_color = centers[dominant_index]
        dominant_colors.append(dominant_color)

    return dominant_colors

def generate_binary_mask(mask_img: Image.Image, class_ids):
    mask_array = np.array(mask_img)
    if isinstance(class_ids, list):
        binary_mask = np.isin(mask_array, class_ids).astype(np.uint8)
    else:
        binary_mask = (mask_array == class_ids).astype(np.uint8)
    return Image.fromarray(binary_mask * 255)

def safe_get(color, index):
    return int(round(color[index])) if index < len(color) else 999

def classify_skin_color(rgb):
    prompt = f"""You are a dermatologist that classifies skin types.

Classify the provided skin color based on RGB values into one of the Fitzpatrick skin types: Type I, Type II, Type III, Type IV, Type V, or Type VI.

Respond with only the type name: Type I, Type II, Type III, Type IV, Type V, or Type VI. Do not include any explanation or extra words.

Examples:
- RGB(255, 224, 189) → Type I
- RGB(241, 194, 125) → Type II
- RGB(224, 172, 105) → Type III
- RGB(198, 134, 66) → Type IV
- RGB(141, 85, 36) → Type V
- RGB(92, 51, 23) → Type VI

Now classify this skin color:
- RGB{rgb} →"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def classify_clothes_color(rgb):
    prompt = f"""You are a fashion expert that classifies clothing colors.

Classify the provided clothing colors based on RGB values into one of the four seasonal color palettes: Spring, Summer, Autumn, or Winter.

Respond with only the season name: Spring, Summer, Autumn, or Winter. Do not include any explanation or extra words.

Examples:
- RGB(255, 223, 186) → Spring
- RGB(176, 224, 230) → Summer
- RGB(153, 101, 21) → Autumn
- RGB(0, 51, 102) → Winter

Now classify this color:
- RGB{rgb} →"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Load the recommendation table
@st.cache_data
def load_recommendation_table():
    return pd.read_csv("csv/clothes_recommendation_table.csv")

# Get recommendations based on predicted skin type
def get_clothing_recommendations(skin_type, df):
    row = df[df["skin_type"] == skin_type]
    if row.empty:
        return []
    return row.iloc[0][["1st", "2nd", "3rd", "4th"]].tolist()

@st.cache_data
def load_score_table():
    return pd.read_csv("csv/clothes_class_score.csv")

def get_scores_for_skin_type(skin_type, df, recommended_classes):
    if skin_type not in df.columns:
        return pd.DataFrame()
    filtered = df[df["clothes_class"].isin(recommended_classes)]
    return filtered[["clothes_class", skin_type]].rename(columns={skin_type: "Score"}).sort_values("Score", ascending=False)

def check_compatibility(skin_type, predicted_clothes_type, recommendation_df):
    recommendations = get_clothing_recommendations(skin_type, recommendation_df)
    top_two = recommendations[:2]  # Only consider top 2 as compatible
    is_compatible = predicted_clothes_type in top_two
    return is_compatible, recommendations


# ========== STREAMLIT UI ==========
st.title("Outfit Color Recommendation")

with st.sidebar:
    st.text("Rayhan Almer Kusumah\n5025211115")

st.markdown("Upload an **image** and. The app will extract dominant face and clothes colors.")

image_file = st.file_uploader("Upload original image", type=["jpg", "jpeg", "png"])

if image_file:
    # Load original image
    pil_image = Image.open(image_file).convert("RGB")

    # Construct SCHP mask path from uploaded image filename
    filename = os.path.splitext(image_file.name)[0]
    schp_mask_path = os.path.join("schp_masks", f"{filename}.png")

    if not os.path.exists(schp_mask_path):
        st.error(f"SCHP mask not found for: {filename}.png in `schp_masks/` folder.")
        st.stop()

    schp_mask = Image.open(schp_mask_path)

    st.divider()

    # Show previews
    st.subheader("Input Preview")

    col = st.columns([1, 2])[0]
    
    with col:
        st.image(pil_image, caption="Original Image", use_container_width=True)


    if st.button("Process and Extract Colors", use_container_width=True):
        with st.spinner("Extracting..."):
            # Generate masks
            face_mask = generate_binary_mask(schp_mask, FACE_CLASS)
            clothes_mask = generate_binary_mask(schp_mask, CLOTHES_CLASSES)

            # Extract dominant colors
            face_color = extract_dominant_color(pil_image, face_mask, 1)[0]
            clothes_color = extract_dominant_color(pil_image, clothes_mask, 1)[0]

            # Convert to RGB + HEX
            face_rgb = tuple(safe_get(face_color, i) for i in range(3))
            clothes_rgb = tuple(safe_get(clothes_color, i) for i in range(3))
            face_hex = "#{:02x}{:02x}{:02x}".format(*face_rgb)
            clothes_hex = "#{:02x}{:02x}{:02x}".format(*clothes_rgb)
            time.sleep(2)
        st.success("Extraction complete!")

        st.divider()

        # Show original image and SCHP mask side by side after processing
        st.subheader("Processed Image & SCHP Mask")
        col1, col2 = st.columns(2)

        with col1:
            st.image(pil_image, caption="Original Image (Input)", use_container_width=True)

        with col2:
            st.image(schp_mask, caption="SCHP Segmentation Mask", use_container_width=True)

        st.divider()


        # Show output
        st.subheader("Dominant Colors")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Skin Color:** {face_rgb}")
            st.markdown(
                f"""
                <div style="width:150px; height:150px; background-color:{face_hex}; 
                            border:0px solid #000; border-radius:8px; margin-bottom:10px;">
                </div>
                """, unsafe_allow_html=True
            )

        with col2:
            st.markdown(f"**Clothes Color:** {clothes_rgb}")
            st.markdown(
                f"""
                <div style="width:150px; height:150px; background-color:{clothes_hex}; 
                            border:0px solid #000; border-radius:8px; margin-bottom:10px;">
                </div>
                """, unsafe_allow_html=True
            )

        st.divider()

        # Optionally show generated masks
        # st.subheader("Generated Masks")
        # col1, col2 = st.columns(2)

        # with col1:
        #     st.image(face_mask, caption="Face Mask", use_container_width=True)

        # with col2:
        #     st.image(clothes_mask, caption="Clothes Mask", use_container_width=True)

        # st.divider()

        recommendation_df = load_recommendation_table()

        face_type = classify_skin_color(face_rgb)
        st.subheader(f"Predicted Skin Type: {face_type}")
        st.image("images/Fitzpatrick-Scale.png", width=500, caption="Fitzpatrick Scale")

        # clothes_type = classify_clothes_color(clothes_rgb)
        # st.subheader(f"Predicted Clothes Type: {clothes_type}")

        # st.divider()

        # # Check compatibility
        # is_compatible, recommended_types = check_compatibility(face_type, clothes_type, recommendation_df)

        # if is_compatible:
        #     st.success(f"The clothing color type **{clothes_type}** is compatible with skin type **{face_type}**.")
        # else:
        #     st.error(f"The clothing color type **{clothes_type}** is NOT recommended for skin type **{face_type}**.")
        #     st.markdown("**Recommended types for your skin:** " + ", ".join(recommended_types[:2]))

        # st.divider()

        clothes_recommendations = get_clothing_recommendations(face_type, recommendation_df)
        score_df = load_score_table()
        score_table = get_scores_for_skin_type(face_type, score_df, clothes_recommendations)
        st.subheader("Recommended Clothing Color Types")

        # Mapping clothes color type to color palette image paths
        palette_image_map = {
            "Spring": "images/Spring.png",
            "Summer": "images/Summer.png",
            "Autumn": "images/Autumn.png",
            "Winter": "images/Winter.png"
        }

        if clothes_recommendations:
            for i, rec in enumerate(clothes_recommendations, start=1):
                col1, col2 = st.columns([1, 2])
                
                # Text column
                with col1:
                    st.markdown(f"**{i}. {rec}**", help="Clothing color category")

                # Image column
                with col2:
                    image_path = palette_image_map.get(rec)
                    if image_path:
                        st.image(image_path, caption=f"{rec} palette", width=400)
                    else:
                        st.warning(f"No image found for {rec}")
        else:
            st.warning("No recommendation found for this skin type.")
        
        st.divider()
        st.subheader("Recommendation Scores")

        if not score_table.empty:
            st.dataframe(score_table.reset_index(drop=True), use_container_width=True)
        else:
            st.warning("No scores available for this skin type.")

else:
    st.info("Please upload your input image.")