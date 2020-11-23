## Notice
- This file can be opened as markdown format `.md`.
- Dependencies
    - Python standard library: `math`, `os`, `pathlib`, `pickle`, `random`, `time`
    - `numpy`
    - `scipy`
    - `matplotlib`
    - `torch`
    - `torchvision`
- HOW TO execute the python script.
    1. Put the following files under **THE SAME DIRECTORY**.
        - `HW2_109062631_Main.py`
        - `HW2_109062631_Module.py`
        - `data.npy`
        - `label.npy`
        - `CAE_optimal.pk`
    2. Enter the following commands at the cmd.exe/terminal.
        - For Microsoft Windows
        	> $ python HW2_109062631_Main.py
        - For Linux
        	> $ python3 HW2_109062631_Main.py
    3. Decide whether enable "load pre-trained model" by enter 'Y' or 'N'.
    4. You would see training loss during each epoch. (if disabling "load pre-trained model")
    5. `gen_data.npy` and `gen_label.npy` are placed at the same directory as python script.
- NOTE: The meaning of dimensions to `gen_data.npy` and `gen_label.npy` are equal to `data.npy` and `label.npy`.