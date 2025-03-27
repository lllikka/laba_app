import streamlit as st

st.set_page_config(page_title="Моё приложение", layout="wide")
st.title("🎉 Моё первое развернутое приложение!")
st.write("Это приложение работает в облаке Streamlit!")

name = st.text_input("Как вас зовут?")
if name:
    st.success(f"Привет, {name}! Рад вас видеть!")
