## Notice
- This file can be opened as markdown format `.md`.
- Dependencies
    - Python standard: `math`, `os`, `pathlib`, `random`, `re`, `struct`, `sys`, `time`, `warnings`
    - `matplotlib`
    - `numpy` and `scipy`
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
    > Epoch=0, total loss=6.5637
    > Epoch=2500, total loss=4.5946
    > Epoch=5000, total loss=3.2162
    > Epoch=7500, total loss=2.2513
    > Epoch=12500, total loss=2.2513
    > Epoch=15000, total loss=1.1032
    > Epoch=17500, total loss=0.7722
    > Epoch=20000, total loss=0.5405
    > 
    > End training.
    > Predict testing data set.
    > 
    > [Result]
    > Accuracy: 93.4322%
    > Total Exe. Seconds: 107.38
