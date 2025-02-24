import os
import time
import cv2
import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

# Define the Predictor class
class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
        )
        return result_img

def get_image_list(path):
    image_names = []
    if os.path.isdir(path):
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in [".jpg", ".jpeg", ".webp", ".bmp", ".png"]:
                    image_names.append(apath)
    else:
        image_names.append(path)
    return image_names

def run_inference_for_image(config_path, model_path, image_path, save_result=False, save_dir='./inference_results'):
    load_config(cfg, config_path)
    logger = Logger(local_rank=0, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device="cuda:0")
    
    image_names = get_image_list(image_path)
    image_names.sort()

    if save_result:
        current_time = time.localtime()
        save_folder = os.path.join(save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank=0, path=save_folder)

    result_images = []
    for image_name in image_names:
        meta, res = predictor.inference(image_name)
        result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)

        if save_result:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)

        result_images.append(result_image)

    return result_images

def extract_license_plate_text(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    ocr_results = ocr.ocr(image, cls=True)

    license_plate_text = ""
    license_plate_box = None

    for line in ocr_results:
        for word_info in line:
            text = word_info[1][0]
            if len(text) >= 5 and len(text) <= 10:
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
    st.title("Car Damage Assessment")

    config_path = 'config/nanodet-plus-m_416-yolo.yml'
    model_path = 'workspace/nanodet-plus-m_416/model_best/model_best.ckpt'
    save_dir = './inference_results'

    image_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if image_file is not None:
        image_path = "./temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.read())

        save_result = st.checkbox("Save Inference Results", value=False)

        with st.spinner("Running inference..."):
            result_images = run_inference_for_image(config_path, model_path, image_path, save_result, save_dir)
            cropped_license_plate, license_plate_text = extract_license_plate_text(result_images[0])

        st.image(result_images[0], caption="Processed Image", use_column_width=True)

        if cropped_license_plate is not None:
            st.image(cropped_license_plate, caption=f"Extracted License Plate: {license_plate_text}", use_column_width=True)
        else:
            st.write("No License Plate Detected")

if __name__ == "__main__":
    main()
