## Notice
- This file can be opened as markdown format `.md`.
- Dependencies
    - Python standard: `math`, `os`, `pathlib`, `random`, `re`, `struct`, `sys`, `time`
    - `numpy` and `scipy`
- If you get some error about `No such file or directory: 'xxxx.pickle'`, please run function `generatePkFromSource()`. After that step, you can disable it.
- HOW TO execute the python script.
    1. Put the following files under **THE SAME DIRECTORY**.
        - `HW1_109062631.py`
        - `t10k-images.idx3-ubyte`
        - `t10k-labels.idx1-ubyte`
        - `train-images.idx3-ubyte`
        - `train-labels.idx1-ubyte`
    2. Enter the following commands at the cmd.exe/terminal.
        - For Microsoft Windows
        	> $ python HW1_109062631.py
        - For Linux
        	> $ python3 HW1_109062631.py
    3. Example print out messages (would not be exactly the same).
    > Open source files.
    > Load data (may take a few minutes).
    > Shuffle data.
    > Generate training set & validation set.
    > Start training a model.
    > 
    > Epoch=0, Total loss=23.29
    > Epoch=1, Total loss=2.30
    > Epoch=2, Total loss=1.15
    > Epoch=3, Total loss=0.77
    > Epoch=4, Total loss=0.58
    > Epoch=5, Total loss=0.46
    > Epoch=6, Total loss=0.38
    > Epoch=7, Total loss=0.33
    > Epoch=8, Total loss=0.29
    > Epoch=9, Total loss=0.26
    > Epoch=10, Total loss=0.23
    > 
    > End training.
    > 
    > Predict testing data set.
    > 
    > [Result]
    > Accuracy on test data: 90.2003%
    > Total Exe. Seconds: 138.04