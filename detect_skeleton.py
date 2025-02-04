import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadRs
from utils.general import (
    check_img_size,
    check_imshow,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import Skeleton3D, output_to_keypoint, plot_skeleton_kpts


def detect():
    source, weights, view_img, imgsz, trace = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.img_size,
        not opt.no_trace,
    )
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )
    realsense = source == "rs"

    # Object to manage the 3D skeleton
    sk = Skeleton3D()

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    trace = False
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    elif realsense:
        dataset = LoadRs(img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        
    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, depth, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression_kpt(pred, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        t3 = time_synchronized()

        # Process detections
        with torch.no_grad():
            pred = output_to_keypoint(pred)
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(pred.shape[0]):
            sk.plot_3D_skeleton(nimg, depth = depth, kpts = pred[idx, 7:].T, steps = 3)

            # Print time (inference + NMS)
            print(f"Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS")

        # Stream results
        if view_img:
            cv2.imshow("Skeletons", nimg)
            cv2.waitKey(1)  # 1 millisecond

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov7-w6-pose.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="rs", help="source")  # file/folder, 0 for webcam, rs for realsense
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
