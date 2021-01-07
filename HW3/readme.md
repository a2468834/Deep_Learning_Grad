## Notice
- This file can be opened as markdown format `.md`.
- Dependencies
    - Python standard: `math`, `os`, `pathlib`, `random`, `re`, `struct`, `sys`, `time`, `warnings`
    - Additional packages: `numpy`, `scipy`, `cv2`, `matplotlib`, `numba`
- HOW TO execute the python script.
    1. Put the following files and directories under **THE SAME DIRECTORY**.
        - `HW3_109062631_Main.py`
        - `HW3_109062631_Module.py`
        - `Data_train` (Extract from `Data.zip` provided by TA)
        - `Data_test` (Extract from `Data.zip` provided by TA)
    2. Enter the following commands at the cmd.exe/terminal.
        - For Microsoft Windows
            - `> $ python HW3_109062631_Main.py`
        - For Linux
            - `> $ python3 HW3_109062631_Main.py`
    3. Example print out messages (may not be exactly the same).
    ```
    Load data from pickles.
    Shuffle data.
    Generate training set & validation set.
    Start training a model.
    
    Epoch:  0  |  Training Loss: 14.6457  |  Validation Loss: 13.7242
    Epoch:  1  |  Training Loss: 11.7109  |  Validation Loss: 10.1303
    Epoch:  2  |  Training Loss:  8.4065  |  Validation Loss:  7.3994
    Epoch:  3  |  Training Loss:  7.5186  |  Validation Loss:  6.6320
    Epoch:  4  |  Training Loss:  7.3287  |  Validation Loss:  6.4477
    Epoch:  5  |  Training Loss:  7.1422  |  Validation Loss:  6.3092
    Epoch:  6  |  Training Loss:  7.1225  |  Validation Loss:  6.2873
    Epoch:  7  |  Training Loss:  7.0507  |  Validation Loss:  6.2332
    Epoch:  8  |  Training Loss:  7.0273  |  Validation Loss:  6.2194
    Epoch:  9  |  Training Loss:  7.0145  |  Validation Loss:  6.2074
    Epoch: 10  |  Training Loss:  7.0170  |  Validation Loss:  6.2085
    
    End training.

    Predict testing data set.

    [Result]
    Accuracy on test data: 95.6113%
    Total execution time: 2725.07 seconds
    ```