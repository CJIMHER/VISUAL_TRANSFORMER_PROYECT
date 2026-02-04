import streamlit as st
from transformers import pipeline
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

pre_process = AutoImageProcessor.from_pretrained('javierrf91/streamlit')


st.title("What is it?")

file_name = st.file_uploader("Upload a hot dog candidate image")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    inputs = pre_process(images=image, return_tensors="pt")
    input_pixels = inputs.pixel_values
    model = AutoModelForImageClassification.from_pretrained('javierrf91/streamlit')
    outputs = model(input_pixels)
    col2.header(model.config.id2label[outputs.logits.argmax(-1)  .item()])