import pytesseract
from PIL import Image
import cv2
import numpy as np
import streamlit as st
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization")

# Streamlit app
st.title('Handwritten Text Recognition with Summarization')

# Upload image
uploaded_image = st.file_uploader("Upload a handwritten image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Load the image using Pillow
    image = Image.open(uploaded_image)
    
    # Convert the image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image (optional, can improve OCR)
    _, processed_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract text using pytesseract
    st.write("Recognizing text...")
    text = pytesseract.image_to_string(Image.fromarray(processed_image))

    # Display the extracted text
    st.write("Extracted Text:")
    st.write(text)

    # Option to summarize the text
    if st.button("Summarize Text"):
        if len(text.strip()) == 0:
            st.warning("No text found to summarize.")
        else:
            # Summarize the text
            summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
            st.write("Summary:")
            st.write(summary)
