import numpy as np
from axengine import InferenceSession

def main():
    model_path = "torch_vision_mobilenet_v2.axmodel"
    
    # Create a simple test image
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = img[None]  # Add batch dimension
    
    # Initialize inference session
    session = InferenceSession(
        path_or_bytes=model_path, 
        providers=['AXCLRTExecutionProvider']
    )
    
    # Debug: Print input and output details
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print("===== INPUT DETAILS =====")
    for idx, input_info in enumerate(inputs):
        print(f"Input {idx}: name={input_info.name}, shape={input_info.shape}")
    
    print("\n===== OUTPUT DETAILS =====")
    for idx, output_info in enumerate(outputs):
        print(f"Output {idx}: name={repr(output_info.name)}, shape={output_info.shape}")
    
    # Get the actual output name from the model
    actual_output_name = outputs[0].name
    print(f"\nActual output name: {repr(actual_output_name)}")
    
    # Run inference with the correct output name
    result = session.run(None, input_feed={"input": img})[0]
    print(f"Output shape: {result.shape}")
    
    return result

if __name__ == "__main__":
    main()