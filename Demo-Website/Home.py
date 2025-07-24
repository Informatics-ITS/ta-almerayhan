import streamlit as st

st.title("Skintone-based Outfit Color Compatibility Identification")

with st.sidebar:
    # st.divider()
    st.text("Rayhan Almer Kusumah\n5025211115")

st.divider()

st.header("What is your Skin Color?")

st.image("images/Fitzpatrick-Scale.png", caption="Fitzpatrick")

st.divider()

st.header("What is Seasonal Color Palette?")

col1, col2 = st.columns(2)

with col1:
    st.image("images/Spring.png", width=300, caption="Spring Color Palette")
    st.image("images/Autumn.png", width=300, caption="Autumn Color Palette")

with col2:
    st.image("images/Summer.png", width=300, caption="Summer Color Palette")
    st.image("images/Winter.png", width=300, caption="Winter Color Palette")

st.divider()
st.header("Bismillah")