import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2
import onnxruntime as ort

# 1. Setup Model (Small/ViT-S for Orin Nano)
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load("depth_anything_v2_metric_vkitti_vits.pth", map_location='cpu'))
model.eval()

model_name = "depth_vits_392x518.onnx"

# 2. Define Custom Resolution (Multiples of 14)
# Width: 518, Height: 392 (Closest to 640x480 aspect)
H, W = 392, 518
dummy_input = torch.randn(1, 3, H, W)

# 3. Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    model_name,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True,    
)

print(f"Exported successfully with shape {H}x{W}")

try:
    session = ort.InferenceSession(model_name)
    print("Model loaded successfully!")
    print("Inputs:", [i.name for i in session.get_inputs()])
    print("Input Shape:", session.get_inputs()[0].shape)
except Exception as e:
    print(f"Load failed: {e}")