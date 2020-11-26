import os
import time

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_with_plugins import TrtYOLO


# Additions for uvc
import uvc
import logging
logging.basicConfig(level=logging.INFO)
dev_list = uvc.device_list()
cap = uvc.Capture(dev_list[0]["uid"])
cap.frame_mode = cap.avaible_modes[2]
cap.get_frame_robust()
time.sleep(1)

def main():
    trt_yolo = TrtYOLO('yolov4-tiny-capillary-apex-detector-416', (416, 416), 1)
    while True:
        img = cap.get_frame_robust().bgr
        boxes, confs, clss = trt_yolo.detect(img, 0.3)
        for bb, cf, cl in zip(boxes, confs, clss):
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            conf_string = f"{cf*100:.0f}%"
            cv2.putText(img, conf_string, (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2) 
			
        cv2.imshow('YOLOv4', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
