from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics import FastSAM
import supervision as sv
import torch
import cv2
import os
import numpy as np
 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_path = r'C:\Users\mende\OneDrive\Desktop\FastSAMAPS\lo3.jpg'
img = cv2.imread(image_path)


# Define an inference source
source = np.array(img)
# Create a FastSAM model
model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt

# Run inference on an image
everything_results = model(source, device=DEVICE, retina_masks=True, imgsz=512, conf=0.4, iou=0.9)

# Prepare a Prompt Process object
prompt_process = FastSAMPrompt(source, everything_results, device=0)

# Everything prompt
ann = prompt_process.everything_prompt()

# # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
# ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

# # Text prompt
# ann = prompt_process.text_prompt(text='a photo of a dog')

# Point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
# ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
prompt_process.plot(annotations=ann, output=(r'C:\Users\mende\OneDrive\Desktop\FastSAMAPS\output'))
