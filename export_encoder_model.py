import torch
import numpy as np
import os 
import onnx

from mobile_sam import sam_model_registry , SamPredictor
from mobile_sam.utils.transforms import ResizeLongestSide
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

image_size = (1024,1024)
checkpoint = 'weights/mobile_sam.pt'
model_type = 'vit_t'
output_path = 'weights/sam_vit_t_encoder-float.onnx'
quantize_output_path = 'weights/sam_vit_t_encoder-quantize-float.onnx'

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')
transform = ResizeLongestSide(sam.image_encoder.img_size)

image = np.zeros((image_size[1] , image_size[0] , 3) , dtype=np.uint8)
input_image = transform.apply_image(image)
input_image_torch = torch.as_tensor(input_image , device='cpu' )
input_image_torch = input_image_torch.permute(2,0,1).contiguous()[None , : , : , :]

class Model(torch.nn.Module):
    def __init__(self , image_size , checkpoint , model_type) -> None:
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device="cpu")
        self.predictor = SamPredictor(self.sam)
        self.image_size = image_size
        
    def forward(self , x):
        self.predictor.set_torch_image(x , (self.image_size))
        return self.predictor.get_image_embedding()

model = Model(image_size , checkpoint , model_type)
model_trace = torch.jit.trace(model , input_image_torch)
torch.onnx.export(
    model_trace , input_image_torch , output_path , 
    input_names=['x'] , output_names=['output'] , opset_version=17
)

onnx_model = onnx.load(output_path)
for input in onnx_model.graph.input:
    input.type.tensor_type.elem_type = 1
onnx.save_model(onnx_model , output_path)

quantize_dynamic(
    model_input=output_path , 
    model_output=quantize_output_path , 
    optimize_model=True , 
    per_channel=False , 
    reduce_range=False , 
    weight_type=QuantType.QUInt8
)
      
onnx_model = onnx.load(quantize_output_path)
for input in onnx_model.graph.input:
    input.type.tensor_type.elem_type = 1
onnx.save_model(onnx_model , quantize_output_path)