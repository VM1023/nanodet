import os
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

# Load Models
@st.cache_resource
def load_yolo_model():
    return YOLO("Model/yolov11n_best.pt")  # Load YOLO Model

@st.cache_resource
def load_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang='en')  # Load PaddleOCR Model

# Initialize models
yolo_model = load_yolo_model()
ocr_model = load_ocr_model()

# Function to perform object detection and crop license plate
def detect_and_crop(image_path):
    # Read the image
    original_image = cv2.imread(image_path)
    results = yolo_model.predict(source=image_path, imgsz=640)
    
    cropped_images = []
    for r in results:
        boxes = r.boxes  # Detected bounding boxes
        
        for box in boxes:
            # Extract coordinates and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls = int(box.cls[0])  # Class of the object
            
            # Check if the detected class is for license plates (assuming class 0 is license plate)
            if cls == 0:
                # Crop the detected region
                cropped_img = original_image[y1:y2, x1:x2]
                cropped_images.append(cropped_img)
                
                # Save temporarily for OCR
                cv2.imwrite("temp_license_plate.jpg", cropped_img)
    return cropped_images

# Function to run OCR and extract text
def run_ocr(image_path):
    ocr_output = ocr_model.ocr(image_path, cls=True)
    license_plate_text = ""
    
    for line in ocr_output:
        for word_info in line:
            text = word_info[1][0]
            if len(text) >= 5 and len(text) <= 10:  # Assuming license plates have 5-10 characters
                license_plate_text = text
                break  # Stop after finding the first valid license plate text
        if license_plate_text:
            break  # Stop if we found a valid text

    return license_plate_text

# Streamlit App Interface
def main():
    st.title("License Plate Detection and OCR")
    st.write("### Upload an Image to Detect and Read License Plate")

    # File uploader
    uploaded_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Save uploaded image temporarily
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Display uploaded image
        st.image(Image.open(image_path), caption="Uploaded Image", use_container_width=True)
        
        # Perform detection and cropping
        st.write("### Detecting License Plate...")
        cropped_images = detect_and_crop(image_path)
        
        if cropped_images:
            st.write("### Cropped License Plate Region(s):")
            for idx, cropped_img in enumerate(cropped_images):
                # Convert to RGB for display
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                st.image(cropped_img_rgb, caption=f"Cropped License Plate {idx + 1}", use_container_width=False)
                
                # Save and run OCR
                cv2.imwrite(f"cropped_license_plate_{idx}.jpg", cropped_img)
                st.write("#### Performing OCR on Cropped License Plate...")
                license_plate_text = run_ocr(f"cropped_license_plate_{idx}.jpg")
                st.write(f"**Detected License Plate Text**: {license_plate_text}")
        else:
            st.write("No License Plate Detected. Please try another image.")

if __name__ == "__main__":
    main()
