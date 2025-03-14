# pyaxengine_axcl_test


### Problem Description





## Environment

・Device: Raspberry Pi 5 + Xinjian M.2 Accelerator Card（AX650N）<br>
・pyaxengine: axengine-0.1.2 (axengine-0.1.2-py3-none-any.whl)<br>
・AXCL-SMI: V2.25.0_20250117163029 <br>

```
raspi5$  cat /proc/device-tree/model
Raspberry Pi 5 Model B Rev 1.0(myenv)

raspi5$ lsb_release -a
No LSB modules are available.
Distributor ID: Debian
Description:    Debian GNU/Linux 12 (bookworm)
Release:        12
Codename:       bookworm

raspi5$ uname -a
Linux raspi5 6.6.74+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.74-1+rpt1 (2025-01-27) aarch64 GNU/Linux

raspi5$ axcl-smi
+------------------------------------------------------------------------------------------------+
| AXCL-SMI  V2.25.0_20250117163029                                Driver  V2.25.0_20250117163029 |
+-----------------------------------------+--------------+---------------------------------------+
| Card  Name                     Firmware | Bus-Id       |                          Memory-Usage |
| Fan   Temp                Pwr:Usage/Cap | CPU      NPU |                             CMM-Usage |
|=========================================+==============+=======================================|
|    0  AX650N                    V2.25.0 | 0000:01:00.0 |                150 MiB /      945 MiB |
|   --   67C                      -- / -- | 1%        0% |                 18 MiB /     2944 MiB |
+-----------------------------------------+--------------+---------------------------------------+

+------------------------------------------------------------------------------------------------+
| Processes:                                                                                     |
| Card      PID  Process Name                                                   NPU Memory Usage |
|================================================================================================|
```

### Steps to reproduce


```
$ python infer_axmodel.py
[INFO] Available providers:  ['AXCLRTExecutionProvider']
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 3.3 3cdead5e
===== INPUT DETAILS =====
Input 0: name=input, shape=[1, 224, 224, 3]

===== OUTPUT DETAILS =====
Output 0: name=<cdata 'char *' 0x26333140>, shape=[1, 1000]

Actual output name: <cdata 'char *' 0x26333140>
Traceback (most recent call last):
  File "/home/nnn/axcl/infer_axmodel.py", line 40, in <module>
    main()
  File "/home/nnn/axcl/infer_axmodel.py", line 34, in main
    result = session.run(None, input_feed={"input": img})[0]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nnn/myenv/lib/python3.11/site-packages/axengine/_session.py", line 118, in run
    return self._sess.run(output_names, input_feed, run_options, shape_group)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nnn/myenv/lib/python3.11/site-packages/axengine/_axclrt.py", line 382, in run
    self._io[0].pOutputs[i].pVirAddr, npy_size
    ^^^^^^^^^^^^^^^^^^^^
AttributeError: cdata 'void *' has no attribute 'pOutputs'
```


infer_axmodel.py

```infer_axmodel
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
```

This figure shows torch_vision_mobilenet_v2.axmodel opened in nectron.
![image](https://github.com/user-attachments/assets/cc0df3d6-6cb6-40f3-9f31-96ea0ade727f)

I put it here.
https://github.com/nnn112358/pyaxengine_axcl_test
