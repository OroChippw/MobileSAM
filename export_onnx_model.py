import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from mobile_sam import sam_model_registry , SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel

def show_mask(mask , ax):
    color = np.array([30 / 255 , 144 / 255 , 255 / 255 , 0.6])
    h , w  = mask.shape[-2:]
    mask_image = mask.reshape(h , w , 1) * color.reshape(1 , 1 , -1)
    ax.imshow(mask_image)

def show_points(coords , labels , ax , marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[: , 0] , pos_points[: , 1] , color='green' , marker='*' , s=marker_size , edgecolor='white' , linewidth=1.25)
    ax.scatter(neg_points[: , 0] , neg_points[: , 1] , color='red' , marker='*' , s=marker_size , edgecolor='white' , linewidth=1.25)

def show_box(box , ax):
    x0 , y0 = box[0] , box[1]
    w , h = box[2] - box[0] , box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0 , y0) ,w ,h , edgecolor='green' , facecolor=(0,0,0,0) , lw=2))

checkpoint = "weights/mobile_sam.pt"
model_type = "vit_t"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
onnx_model_path = "sam_vit_t.onnx"

onnx_model_quantized_path = "sam_vit_t-quantized.onnx"

