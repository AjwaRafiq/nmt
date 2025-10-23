import streamlit as st
import torch

st.title("Neural Machine Translation (NMT) Demo")

st.write("Upload a sentence and see the translation result!")

text = st.text_input("Enter text:")

if text:
    st.success(f"Translated text: [Your model will translate '{text}' here]")
