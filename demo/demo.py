import os
import time
import cv2
import torch
import argparse
import multiprocessing
import sys
import numpy as np
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model config file path")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--path", default="./demo", help="Path to image or folder")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of CUDA")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    return parser.parse_args()

class Predictor:
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=self.device)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(self.device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        if not os.path.exists(img):
            print(f"Error: Image path '{img}' does not exist.")
            return None, None
        
        img_data = cv2.imread(img)
        if img_data is None:
            print(f"Error: Failed to load image '{img}'.")
            return None, None
        
        img_info = {"id": 0, "file_name": os.path.basename(img), "height": img_data.shape[0], "width": img_data.shape[1]}
        meta = {"img_info": img_info, "raw_img": img_data, "img": img_data}
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        
        with torch.no_grad():
            results = self.model.inference(meta)
        
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres=0.6, timeout_seconds=300):
        try:
            result_img = self.model.head.show_result(meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False)
            
            cv2.imshow("Detection Result", result_img)
            start_time = time.time()
            while True:
                if cv2.waitKey(100) != -1 or time.time() - start_time > timeout_seconds:
                    break
                time.sleep(0.1)
            cv2.destroyAllWindows()
            return result_img
        except Exception as e:
            print(f"Visualization error: {e}")
            return None

def run_inference_for_image(config_path, model_path, image_path, save_result=False, save_dir='./demo_results'):
    load_config(cfg, config_path)
    logger = Logger(local_rank=0, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device="cuda:0")
    image_names = get_image_list(image_path)
    image_names.sort()
    
    if save_result:
        mkdir(local_rank=0, path=save_dir)
    
    result_images = []
    for image_name in image_names:
        meta, res = predictor.inference(image_name)
        if meta and res:
            result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
            if save_result:
                save_file_name = os.path.join(save_dir, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            result_images.append(result_image)
    return result_images

def get_image_list(path):
    image_names = []
    if os.path.isdir(path):
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_names.append(apath)
    else:
        image_names.append(path)
    return image_names

def run_detection(args):
    try:
        device = "cpu" if args.cpu else "cuda:0"
        load_config(cfg, args.config)
        logger = Logger(0, use_tensorboard=False)
        predictor = Predictor(cfg, args.model, logger, device=device)
        
        if os.path.isfile(args.path):
            print(f"Running inference on {args.path}...")
            meta, res = predictor.inference(args.path)
            if meta and res:
                predictor.visualize(res[0], meta, cfg.class_names, 0.6, args.timeout)
    except Exception as e:
        print(f"Error in detection process: {e}")
    finally:
        sys.exit(0)

def monitor_process(timeout_seconds):
    start_time = time.time()
    while time.time() - start_time <= timeout_seconds + 30:
        time.sleep(0.5)
    os._exit(0)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    
    print(f"Starting object detection with {args.timeout} second timeout...")
    
    detection_p = multiprocessing.Process(target=run_detection, args=(args,))
    detection_p.start()
    
    monitor_p = multiprocessing.Process(target=monitor_process, args=(args.timeout,))
    monitor_p.daemon = True
    monitor_p.start()
    
    detection_p.join(timeout=args.timeout + 60)
    if detection_p.is_alive():
        detection_p.terminate()
    
    print("Object detection completed.")
    sys.exit(0)
