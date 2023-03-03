# -*- coding: UTF-8 -*-
from os import path as os_path
import cv2
from cv_bridge import CvBridge as cvB
from yolov5.detect import run as yolo
from yolov5.utils.augmentations import letterbox
import pyrealsense2 as rs
import json
import torch
import torch.backends.cudnn as cudnn
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size=640 # Size to convert the image to the proper yolo size
    stride = 32 # Parameter for yolo
    positions = []
    # 3D Pose estimation
    A= np.array([
        [ 906.805908203125,0.0,645.4268798828125],
        [ 0.0, 905.5286254882812, 375.8621520996094],
        [ 0.0, 0.0, 1.0 ]
    ])
    inv = np.linalg.inv(A)
    pix = np.array([0,0,0])
    P = np.array([0,0,0])
    # We start the camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # Start streaming
    pipeline.start(config)
    cont = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth = frames.get_depth_frame()
            # Convert images to numpy arrays
            orgimg = np.asanyarray(color_frame.get_data())
            im0 = orgimg.copy() # Image in the original format
            cv_image = letterbox(orgimg, img_size, stride=stride)[0].transpose(2,0,1) # Converting image to yolo format
            pred = yolo(im = cv_image, im0=im0)
            if len(pred):
                im0 = cv2.rectangle(im0, (int(pred[0] - pred[2]/2), int(pred[1] - pred[3]/2)), (int(pred[0] + pred[2]/2), int(pred[1] + pred[3]/2)), (0, 255, 0), 2)
                dist = depth.get_distance(int(pred[0]), int(pred[1]))
                pix[0]=pred[0]
                pix[1]=pred[1]
                pix[2]=1
                P = np.matmul(inv,pix)
                P[0] = dist * P[0]
                P[1] = dist * P[1]
                P[2] = dist
                positions.append(P.tolist())
            else:
                positions.append(P.tolist())
            cv2.imshow("Person",im0)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        dictionary ={
            "positions" : positions,
        }
        with open("positions.json", "w") as outfile:
            json.dump(dictionary, outfile)
    
