import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from mobile_sam import sam_model_registry , SamAutomaticMaskGenerator , SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image_path = r"images/picture2.jpg"
image = cv2.imread(image_path , cv2.COLOR_BGR2RGB)

sam_checkpoint = "./weights/mobile_sam.pt"
model_type = "vit_t"

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(sam)

time_start = time.time()
masks = mask_generator.generate(image)
time_end = time.time()
time_cost = time_end - time_start
print(f"mask_generator Generate mask time cost {time_cost}" , "s")
print("len(masks) : " , len(masks))
print("masks[0].keys() : " , masks[0].keys())

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam , 
    points_per_side=32 , 
    pred_iou_thresh=0.86 , 
    stability_score_thresh=0.92 , 
    crop_n_layers=1,
    crop_n_points_downscale_factor=2 ,
    min_mask_region_area=100
)

time_start = time.time()
masks2 = mask_generator_2.generate(image)
time_end = time.time()
time_cost = time_end - time_start
print(f"mask_generator_2 Generate mask time cost {time_cost}" , "s")
print("len(masks2) : " , len(masks2))

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks2)
# plt.axis('off')
# plt.show() 

