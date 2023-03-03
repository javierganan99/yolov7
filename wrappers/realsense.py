import pyrealsense2 as rs
import numpy as np
import cv2
from threading import Thread
import time

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class LoadRs:  # Load frames for the realsense
    def __init__(self, img_size=640, stride=32, fps = 30):
        self.init_pipeline(fps)
        self.img_size = img_size
        self.stride = stride
        self.imgs = [None]
        # Check first image
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        self.imgs[0] = np.asanyarray(color_frame.get_data())
        thread = Thread(target=self.update, daemon=True)
        thread.start()
        print("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def init_pipeline(self, fps):
        self.fps = fps
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break

        assert found_rgb, "Depth camera with Color sensor not found"

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, self.fps)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)

        # Start streaming
        self.pipeline.start(config)

    def update(self):
        # Read next stream frame in a daemon thread
        while True:
            # Get color_frame
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # Convert image to numpy array
            self.imgs[0] = np.asanyarray(color_frame.get_data())
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        
        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return 0, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
    
    def __del__(self):
        self.pipeline.stop()

    
if __name__ == "__main__":
    lrs = LoadRs()
    try:
        for i in lrs:
            _,img,_,_ = i
            cv2.imshow("Image", img.T[:,:,:,0])
            cv2.waitKey(1)
    finally:
        del rs