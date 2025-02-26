import os
import cv2
import streamlit as st
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_license_plate_text(image):
    ocr_results = ocr.ocr(image, cls=True)

    license_plate_text = ""
    license_plate_box = None

    for line in ocr_results:
        for word_info in line:
            text = word_info[1][0]
            if len(text) >= 5 and len(text) <= 10:  # Length criteria for license plates
                license_plate_text = text
                license_plate_box = word_info[0]

    if license_plate_box is not None:
        points = np.array(license_plate_box, dtype=np.int32)
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        cropped_license_plate = image[y_min:y_max, x_min:x_max]
        return cropped_license_plate, license_plate_text
    return None, None

# Streamlit UI
def main():
    st.title("License Plate Detection")

    image_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if image_file is not None:
        image_path = "./temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.read())

        # Read the uploaded image
        image = cv2.imread(image_path)

        with st.spinner("Extracting license plate..."):
            cropped_license_plate, license_plate_text = extract_license_plate_text(image)

        st.image(image, caption="Original Image", use_column_width=True)

        if cropped_license_plate is not None:
            st.image(cropped_license_plate, caption=f"Extracted License Plate: {license_plate_text}", use_column_width=True)
            st.markdown(f"<h1 style='text-align: center; color: green;'>{license_plate_text}</h1>", unsafe_allow_html=True)
        else:
            st.write("No License Plate Detected")

if __name__ == "__main__":
    main()
