# MangaLineExtraction_PyTorch
# https://github.com/ljsabc/MangaLineExtraction_PyTorch

#NOTE: This preprocessor is designed to work with lineart_anime ControlNet

import torch
import numpy as np
import os
import cv2
from einops import rearrange
from comfy_controlnet_preprocessors.util import annotator_ckpts_path
import model_management
from .model_torch import res_skip

class MangaLineExtractor:
    def __init__(self):
        remote_model_path = "https://github.com/ljsabc/MangaLineExtraction_PyTorch/releases/download/v1/erika.pth"
        modelpath = os.path.join(annotator_ckpts_path, "erika.pth")
        if not os.path.exists(modelpath):
            from comfy_controlnet_preprocessors.util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        net = res_skip()
        net.load_state_dict(torch.load(modelpath))
        net = net.to(model_management.get_torch_device())
        net.eval()
        self.model = net
    def __call__(self, input_image):
        H, W, C = input_image.shape
        Hn = 256 * int(np.ceil(float(H) / 256.0))
        Wn = 256 * int(np.ceil(float(W) / 256.0))
        img = cv2.resize(input_image, (Wn, Hn), interpolation=cv2.INTER_CUBIC)
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(model_management.get_torch_device())
            image_feed = image_feed / 127.5 - 1.0
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')

            line = self.model(image_feed)[0, 0] * 127.5 + 127.5
            line = line.cpu().numpy()

            line = cv2.resize(line, (W, H), interpolation=cv2.INTER_CUBIC)
            line = line.clip(0, 255).astype(np.uint8)
            return line