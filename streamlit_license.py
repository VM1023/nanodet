import os
import time
import cv2
import torch
import numpy as np
import streamlit as st
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import re
from PIL import Image

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
CONFIDENCE_THRESHOLD = 0.50

# Fixed paths
CONFIG_PATH = 'config/nanodet-plus-m_416-yolo.yml'
MODEL_PATH = 'workspace/nanodet-plus-m_416/model_best/nanodet_model_best.pth'
SAVE_DIR = 'Results'
SAVE_RESULTS = True

def initialize_azure_client():
    api_key = st.secrets["AZURE_API_KEY"]
    endpoint = st.secrets["AZURE_ENDPOINT"]
    if not api_key or not endpoint:
        raise ValueError("Azure API Key or Endpoint is missing!")
    return ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))

def extract_license_plate_text(ocr_text):
    if not ocr_text or "Error" in ocr_text or "No text" in ocr_text:
        return ocr_text
    lines = ocr_text.strip().split('\n')
    cleaned_lines = []
    plate_pattern = r'([A-Z]{2}\s*[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{4})'
    for line in lines:
        match = re.search(plate_pattern, line)
        if match:
            cleaned_lines.append(match.group(1).strip())
    return ' '.join(cleaned_lines) if cleaned_lines else "No valid license plate format detected"

def extract_text_from_image(client, image):
    try:
        _, buffer = cv2.imencode(".jpg", image)
        image_stream = BytesIO(buffer.tobytes())
        response = client.read_in_stream(image_stream, raw=True)
        operation_id = response.headers.get("Operation-Location").split("/")[-1]
        max_wait_time = 30
        start_time = time.time()
        while True:
            result = client.get_read_result(operation_id)
            if result.status in [OperationStatusCodes.succeeded, OperationStatusCodes.failed]:
                break
            if time.time() - start_time > max_wait_time:
                return "Error: OCR processing timed out."
            time.sleep(2)
        extracted_text = "\n".join(line.text for page in result.analyze_result.read_results for line in page.lines) if result.status == OperationStatusCodes.succeeded else "No text found."
        return extract_license_plate_text(extracted_text) if extracted_text else extracted_text
    except Exception as e:
        return f"Error: {str(e)}"

def crop_detected_objects(image, dets, class_names, score_thres=0.5):
    return [(image[int(y0):int(y1), int(x0):int(x1)], class_names[label], score, (x0, y0, x1, y1))
            for label in range(len(class_names))
            for bbox in dets[label] if (score := bbox[-1]) > score_thres
            for x0, y0, x1, y1 in [bbox[:4]]]

def run_inference_for_image(config_path, model_path, image_path, azure_client=None, save_result=False, save_dir='./demo_results', score_thres=0.5, progress_bar=None):
    load_config(cfg, config_path)
    logger = Logger(local_rank=0, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device="cuda:0" if torch.cuda.is_available() else "cpu", score_thres=score_thres)
    image_names = sorted(get_image_list(image_path))
    
    save_folder = os.path.join(save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) if save_result else None
    mkdir(local_rank=0, path=save_folder) if save_result else None
    crop_folder = os.path.join(save_folder, "crops") if save_result else None
    mkdir(local_rank=0, path=crop_folder) if save_result else None

    all_cropped_objects = []
    result_images = []

    for i, image_name in enumerate(image_names):
        if progress_bar:
            progress_bar.progress((i + 1) / len(image_names))
            
        meta, res = predictor.inference(image_name)
        raw_img = meta["raw_img"][0]
        result_image = predictor.visualize(res[0], meta, cfg.class_names)
        cropped_objects = crop_detected_objects(raw_img, res[0], cfg.class_names, score_thres)

        if azure_client:
            cropped_objects_with_ocr = []
            if progress_bar:
                ocr_progress_text = st.empty()
            
            for idx, (crop, class_name, score, bbox) in enumerate(cropped_objects):
                if progress_bar:
                    ocr_progress_text.text(f"Processing OCR for object {idx+1}/{len(cropped_objects)}...")
                ocr_text = extract_text_from_image(azure_client, crop)
                cropped_objects_with_ocr.append((crop, class_name, score, bbox, ocr_text))
                
            all_cropped_objects.append((os.path.basename(image_name), cropped_objects_with_ocr))
            if progress_bar and 'ocr_progress_text' in locals():
                ocr_progress_text.empty()
        else:
            all_cropped_objects.append((os.path.basename(image_name), cropped_objects))

        if save_result:
            cv2.imwrite(os.path.join(save_folder, os.path.basename(image_name)), result_image)
            for i, item in enumerate(all_cropped_objects[-1][1]):
                crop, class_name, score, _, ocr_text = item if azure_client else item + ("OCR not performed",)
                with open(os.path.join(crop_folder, f"{os.path.splitext(os.path.basename(image_name))[0]}_{class_name}_{i}_{score:.2f}.txt"), 'w') as f:
                    f.write(ocr_text)
                cv2.imwrite(os.path.join(crop_folder, f"{os.path.splitext(os.path.basename(image_name))[0]}_{class_name}_{i}_{score:.2f}.jpg"), crop)

        result_images.append(result_image)

    return all_cropped_objects, raw_img, save_folder if save_result else None

def run_inference_on_image_array(config_path, model_path, image_array, img_name="captured_image.jpg", azure_client=None, save_result=False, save_dir='./demo_results', score_thres=0.5, progress_bar=None):
    load_config(cfg, config_path)
    logger = Logger(local_rank=0, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device="cuda:0" if torch.cuda.is_available() else "cpu", score_thres=score_thres)
    
    save_folder = os.path.join(save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) if save_result else None
    mkdir(local_rank=0, path=save_folder) if save_result else None
    crop_folder = os.path.join(save_folder, "crops") if save_result else None
    mkdir(local_rank=0, path=crop_folder) if save_result else None

    # Process the image
    meta, res = predictor.inference(image_array)
    raw_img = meta["raw_img"][0]
    result_image = predictor.visualize(res[0], meta, cfg.class_names)
    cropped_objects = crop_detected_objects(raw_img, res[0], cfg.class_names, score_thres)

    if azure_client:
        cropped_objects_with_ocr = []
        if progress_bar:
            ocr_progress_text = st.empty()
        
        for idx, (crop, class_name, score, bbox) in enumerate(cropped_objects):
            if progress_bar:
                ocr_progress_text.text(f"Processing OCR for object {idx+1}/{len(cropped_objects)}...")
            ocr_text = extract_text_from_image(azure_client, crop)
            cropped_objects_with_ocr.append((crop, class_name, score, bbox, ocr_text))
            
        all_cropped_objects = [(img_name, cropped_objects_with_ocr)]
        if progress_bar and 'ocr_progress_text' in locals():
            ocr_progress_text.empty()
    else:
        all_cropped_objects = [(img_name, cropped_objects)]

    if save_result:
        cv2.imwrite(os.path.join(save_folder, img_name), result_image)
        for i, item in enumerate(all_cropped_objects[0][1]):
            crop, class_name, score, _, ocr_text = item if azure_client else item + ("OCR not performed",)
            with open(os.path.join(crop_folder, f"{os.path.splitext(img_name)[0]}_{class_name}_{i}_{score:.2f}.txt"), 'w') as f:
                f.write(ocr_text)
            cv2.imwrite(os.path.join(crop_folder, f"{os.path.splitext(img_name)[0]}_{class_name}_{i}_{score:.2f}.jpg"), crop)

    return all_cropped_objects, raw_img, save_folder if save_result else None

class Predictor:
    def __init__(self, cfg, model_path, logger, device="cuda:0", score_thres=0.5):
        self.cfg = cfg
        self.device = device
        self.score_thres = score_thres
        model = build_model(cfg.model)
        load_model_weight(model, torch.load(model_path, map_location=lambda storage, loc: storage), logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0, "file_name": os.path.basename(img) if isinstance(img, str) else None}
        img = cv2.imread(img) if isinstance(img, str) else img
        img_info.update({"height": img.shape[0], "width": img.shape[1]})
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            return meta, self.model.inference(meta)

    def visualize(self, dets, meta, class_names):
        return self.model.head.show_result(meta["raw_img"][0], dets, class_names, score_thres=self.score_thres, show=False)

def get_image_list(path):
    return [os.path.join(maindir, filename) for maindir, _, filelist in os.walk(path) for filename in filelist if os.path.splitext(filename)[1] in image_ext] if os.path.isdir(path) else [path]

def display_side_by_side(original_image, cropped_objects):
    if not cropped_objects:
        st.warning("No objects detected!")
        return
    
    # Display original image on the left
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Display detected objects on the right
    with col2:
        st.subheader("Detected Objects with OCR")
        if len(cropped_objects) > 0:
            # Show first detected object
            item = cropped_objects[0]
            crop, class_name, score, bbox, ocr_text = item if len(item) == 5 else (*item, "OCR not performed")
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"{class_name}: {score:.2f}", use_column_width=True)
            st.text(f"Size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}px")
            st.text_area("OCR Result", ocr_text, height=100, key="first_ocr")
            
            # If there are more objects, create an expander to show them
            if len(cropped_objects) > 1:
                with st.expander(f"Show {len(cropped_objects)-1} more detected objects"):
                    for i, item in enumerate(cropped_objects[1:], 1):
                        st.divider()
                        crop, class_name, score, bbox, ocr_text = item if len(item) == 5 else (*item, "OCR not performed")
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"{class_name}: {score:.2f}", use_column_width=True)
                        st.text(f"Size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}px")
                        st.text_area("OCR Result", ocr_text, height=100, key=f"ocr_{i}")
        else:
            st.info("No objects detected at the current confidence threshold.")

def main():
    st.set_page_config(page_title="NanoDet Object Detection with OCR", layout="wide")
    
    st.title("License Plate Detection with OCR")
    st.write("Upload an image or use your camera to detect objects and perform OCR on license plates")
    
    # Simplified sidebar with only confidence threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=CONFIDENCE_THRESHOLD, step=0.05)
    
    # Use Azure OCR automatically
    try:
        azure_client = initialize_azure_client()
    except Exception as e:
        st.error(f"Azure OCR client initialization failed: {str(e)}")
        azure_client = None

    # Create tabs for image source selection
    tab1, tab2 = st.tabs(["Upload Image", "Take Picture"])
    
    # Results placeholder (outside tabs to maintain visibility)
    results_placeholder = st.empty()
    
    # File upload tab
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=list(ext.replace(".", "") for ext in image_ext))
        
        if uploaded_file is not None:
            # Create a container for results
            with results_placeholder.container():
                # Save uploaded file temporarily
                temp_file_path = os.path.join(os.getcwd(), "temp_upload", uploaded_file.name)
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Run inference
                    with st.spinner("Processing image..."):
                        progress_bar = st.progress(0)
                        all_cropped_objects, original_image, save_path = run_inference_for_image(
                            CONFIG_PATH, MODEL_PATH, temp_file_path,
                            azure_client=azure_client,
                            save_result=SAVE_RESULTS, save_dir=SAVE_DIR,
                            score_thres=confidence_threshold,
                            progress_bar=progress_bar
                        )
                    
                    # Display original image and cropped objects side by side
                    if all_cropped_objects:
                        image_name, cropped_objects = all_cropped_objects[0]
                        display_side_by_side(original_image, cropped_objects)
                    else:
                        st.warning("No objects detected.")
                        
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                
                # Clean up
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
    # Camera tab
    with tab2:
        st.write("Take a picture with your camera")
        
        # Camera input widget
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            # Create a container for results
            with results_placeholder.container():
                try:
                    # Convert the camera image to OpenCV format
                    img = Image.open(camera_image)
                    img_array = np.array(img)
                    # Convert RGB to BGR (OpenCV format)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Run inference
                    with st.spinner("Processing image..."):
                        progress_bar = st.progress(0)
                        all_cropped_objects, original_image, save_path = run_inference_on_image_array(
                            CONFIG_PATH, MODEL_PATH, img_array,
                            img_name="camera_image.jpg",
                            azure_client=azure_client,
                            save_result=SAVE_RESULTS, save_dir=SAVE_DIR,
                            score_thres=confidence_threshold,
                            progress_bar=progress_bar
                        )
                    
                    # Display original image and cropped objects side by side
                    if all_cropped_objects:
                        image_name, cropped_objects = all_cropped_objects[0]
                        display_side_by_side(original_image, cropped_objects)
                    else:
                        st.warning("No objects detected.")
                        
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
