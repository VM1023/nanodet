import os
import time
import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Define a method to load the config, model, and run inference for image.
def run_inference_for_image(config_path, model_path, image_path, save_result=False, save_dir='./demo_results'):
    # Load the configuration file
    load_config(cfg, config_path)

    # Initialize logger (can be a placeholder since we're not using TensorBoard in Jupyter)
    logger = Logger(local_rank=0, use_tensorboard=False)

    # Initialize the predictor (the model)
    predictor = Predictor(cfg, model_path, logger, device="cuda:0")
    
    # Get the image list (this can be a single image or a folder)
    image_names = get_image_list(image_path)
    image_names.sort()

    # Create a directory to save the results
    current_time = time.localtime()
    if save_result:
        save_folder = os.path.join(save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank=0, path=save_folder)

    # Process each image
    result_images = []
    for image_name in image_names:
        meta, res = predictor.inference(image_name)
        result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)

        # Save the result image if specified
        if save_result:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)

        # Append the result image to the list to display later
        result_images.append(result_image)

    return result_images


# Define the predictor class (same as before, no changes)
class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
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
                if ext in image_ext:
                    image_names.append(apath)
    else:
        image_names.append(path)
    return image_names
