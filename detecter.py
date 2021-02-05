from pathlib import Path

import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords,set_logging, increment_path
from utils.torch_utils import select_device
from config import *

class Detecter():
    def __init__(self):
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz = CONF_SOURCE,CONF_WEIGHT, CONF_VIEW_IMG,CONF_SAVE_TXT,CONF_IMG_SIZE

        # Directories
        self.save_dir = Path(increment_path(Path(CONF_PROJECT) / CONF_NAME))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(CONF_DEVICE)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        self.save_img = False

    def detect(self):
        counter=0

        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        for path, img, im0s, _ in dataset:

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img)[0]
            pred = non_max_suppression(prediction=pred)

            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        if (int(c) == 0):
                            counter+=n.item()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
        return counter
