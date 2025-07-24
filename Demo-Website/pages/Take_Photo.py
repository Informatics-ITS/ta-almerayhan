import streamlit as st

st.title("Take Photo")

with st.sidebar:
    st.text("Rayhan Almer Kusumah\n5025211115")


enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    st.image(picture)