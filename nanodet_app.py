import os
import cv2
import streamlit as st
import numpy as np
import re
from paddleocr import PaddleOCR

# Initialize PaddleOCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_image(image):
    """Preprocess image to enhance license plate visibility."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Use adaptive thresholding instead of Canny edges
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def detect_license_plate(image):
    """Detects and extracts the license plate from an image."""
    processed = preprocess_image(image)

    # Find external contours (better for license plate detection)
    cnts, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # Keep the largest 10 contours

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select contour with 4 corners
            x, y, w, h = cv2.boundingRect(c)

            # Filter based on reasonable plate dimensions
            aspect_ratio = w / float(h)
            if 2 <= aspect_ratio <= 6:  # License plates typically have a width-to-height ratio between 2:1 and 6:1
                cropped_plate = image[y:y+h, x:x+w]

                # Resize to improve OCR accuracy
                cropped_plate = cv2.resize(cropped_plate, (300, 100))

                return cropped_plate  # Return cropped license plate

    return None  # Return None if no plate found

def extract_license_plate_text(image):
    """Extracts text from the detected license plate using PaddleOCR."""
    if image is None:
        return None, None

    # Convert image to RGB (PaddleOCR requires RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run OCR on the cropped license plate
    ocr_results = ocr.ocr(image_rgb, cls=True)

    # Process OCR results
    detected_text = None
    for line in ocr_results:
        for word_info in line:
            text = word_info[1][0]

            # Regex to filter only valid license plate patterns (letters and numbers)
            if re.match(r"^[A-Z0-9]{5,10}$", text):  
                detected_text = text
                break

    return image, detected_text

# Streamlit UI
def main():
    st.title("🚗 License Plate Recognition with OCR")
    image_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if image_file is not None:
        # Save the uploaded image temporarily
        image_path = "./temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.read())

        # Read the uploaded image
        image = cv2.imread(image_path)

        if image is None:
            st.error("⚠ Error reading the image. Please upload a valid image file.")
            return

        with st.spinner("🔍 Detecting license plate..."):
            cropped_plate = detect_license_plate(image)

        if cropped_plate is not None:
            with st.spinner("📖 Performing OCR..."):
                cropped_license_plate, license_plate_text = extract_license_plate_text(cropped_plate)

            if license_plate_text:
                st.image(cropped_license_plate, caption=f"✅ Extracted License Plate: {license_plate_text}", use_column_width=True)
                st.markdown(f"<h1 style='text-align: center; color: green;'>{license_plate_text}</h1>", unsafe_allow_html=True)
            else:
                st.image(cropped_license_plate, caption="❌ No Valid License Plate Text Detected", use_column_width=True)

        else:
            st.error("❌ No License Plate Detected")

if __name__ == "__main__":
    main()
