import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2
import onnxruntime as ort
import cv2
import numpy as np

# --- Configure variables ---
model_path = "depth_anything_v2_metric_vkitti_vits.pth"
onnx_export_name = "depth_vits_392x518.onnx"
image_paths = ["../example_images/close_distance.jpg", "../example_images/larger_distance.jpg"]

# 1 --- Load original model for onnx export ---
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 2 --- Create onnx export ----
# Define Custom Resolution (Multiples of 14)
# Width: 518, Height: 392 (Closest to 640x480 aspect)

H, W = 392, 518
dummy_input = torch.randn(1, 3, H, W)

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_export_name,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True,    
)

print(f"Exported successfully with shape {H}x{W}")

# 3 --- Validate the onnx export dimensions ---
try:
    session = ort.InferenceSession(onnx_export_name)
    print("Model loaded successfully!")
    print("Inputs:", [i.name for i in session.get_inputs()])
    print("Input Shape:", session.get_inputs()[0].shape)
except Exception as e:
    print(f"Load failed: {e}")

# 4 --- Load onnx model and Pytorch model on GPU
onnx_session = ort.InferenceSession(onnx_export_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(model_path, map_location=device)) # Load to correct device
model.to(device)
model.eval()

# 5 --- Load and prepare a sample images for the prediction comparison
# Depth anything expects the images in RGB order
def compare(image_path):
    img_bgr = cv2.imread(image_path)
    height, width, _ = img_bgr.shape

    # --- Pytorch prediction ---
    # Infer_image contains the model preparation steps, it contains helper wrappers
    with torch.no_grad():
        pytorch_depth = model.infer_image(img_bgr, input_size=518)

    # --- ONNX prediction ---
    # This path has to do the manual preprocessing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (518, 392)) # W, H 
    img = img.astype(np.float32) / 255.0 # To float and normalize
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # Normalize to imagenet
    img_input = img.transpose(2, 0, 1)[None] # Transpose to NCHW
    img_input = img_input.astype(np.float32)
    onnx_output = onnx_session.run(None, {'input': img_input})[0].squeeze()

    onnx_depth = cv2.resize(onnx_output, (width, height))

    # Create pixel difference
    error_map = np.abs(pytorch_depth - onnx_depth)
    mae = np.mean(error_map)

    # Create 
    ratio = (onnx_depth + 1e-6) / (pytorch_depth + 1e-6) # 0.5 is 100% closer, 1 is perfect match and 2.0 is double the distance.
    log_ratio = np.clip(np.log2(ratio), -1, 1) # Converts 0.5 into -1, 1 into 0 and 2.0 into 1. Also clips between -1 and 1
    ratio_int_8 = ((log_ratio+1)/2 * 255).astype(np.uint8) # Remap from -1 and 1 to 0 and 255
    distortion_map = cv2.applyColorMap(ratio_int_8, cv2.COLORMAP_JET)

    def colorize(depth):
        # Normalize to 0-1 for colormap
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        # Apply colormap and convert to BGR for OpenCV
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        return depth_colored

    vis_pytorch = colorize(pytorch_depth)
    vis_onnx = colorize(onnx_depth)

    # Create the white vertical divider (50 pixels wide)
    split_region = np.ones((height, 50, 3), dtype=np.uint8) * 255

    def create_legend(height):
        # 1. Create a 256-pixel gradient (0 to 255)
        # Reverse it so 255 (Red/Double) is at the top
        gradient = np.linspace(255, 0, 256).astype(np.uint8).reshape(-1, 1)
        
        # 2. Apply the same JET colormap
        legend_colors = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        
        # 3. Resize to match the image height and give it some width
        legend_bar = cv2.resize(legend_colors, (50, height), interpolation=cv2.INTER_NEAREST)
        
        # 4. Add Text Labels (Top, Middle, Bottom)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(legend_bar, '2.0x', (5, 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(legend_bar, '1.0x', (5, height//2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(legend_bar, '0.5x', (5, height-15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return legend_bar
    
    combined = cv2.hconcat([img_bgr, split_region, vis_pytorch, split_region, vis_onnx, split_region, distortion_map, create_legend(height)])
    
    cv2.imwrite(f'example_{os.path.basename(image_path)}', combined)
    print(f"Saved comparison to example_{os.path.basename(image_path)}, the mae is {mae:.4f}")
   

for image_path in image_paths:
    compare(image_path)