# pyaxengine_axcl_test

## Environment
- Hardware: Raspberry Pi 5 + Xinjian M.2 Accelerator Card (AX650N)
- Software: pyaxengine-0.1.2 (axengine-0.1.2-py3-none-any.whl)
- AXCL-SMI: V2.25.0_20250117163029

## Problem Description
The issue occurs when trying to run inference with a MobileNetV2 model (torch_vision_mobilenet_v2.axmodel) using the AX650N accelerator:

1. The output name from the model is displayed as a C-style pointer (`<cdata 'char *' 0x26333140>`) instead of a proper Python string.

2. When executing `session.run(None, input_feed={"input": img})[0]`, the following error occurs:
   ```
   AttributeError: cdata 'void *' has no attribute 'pOutputs'
   ```



