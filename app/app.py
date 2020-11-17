import streamlit as st
import io
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Facial emotion analytics")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
