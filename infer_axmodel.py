import cv2
import numpy as np
from axengine import InferenceSession

def main():
    img_path = "input.jpg"
    model_path = "depth_anything_v2_vits_ax650.axmodel"
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (518, 518))
    img = img[None]  # Add batch dimension
    
    # Initialize inference session and run model
    session = InferenceSession(
        path_or_bytes=model_path, 
        providers=['AXCLRTExecutionProvider']
    )

    print("Input names:", [input.name for input in session.get_inputs()])
    print("Input shape:", [input.shape for input in session.get_inputs()])

    print("Output names:", [output.name for output in session.get_outputs()])
    print("Output shape:", [output.shape for output in session.get_outputs()])

    # Run inference
    depth = session.run(output_names=["output"], input_feed={"input":img})[0]

    # Print shape of output (optional)
    print(f"Depth map shape: {depth[0].shape}")
    
    return depth

if __name__ == "__main__":
    main()