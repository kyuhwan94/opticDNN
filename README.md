# atomDL
Reconstructing atomic absorption signals using deep learning

1. ./20241231_full: Contains example data that can be used to test the training protocol
2. ./configs: Contains an example configuration that can used for training.
3. ./src: Contains source code for training
4. ./src/run.sh: Shell script used to execute train.py
5. ./src/train.py: Python code that constructs the pipeline of the DNN + train via back propagation
6. ./src/predict.py: Python code that can be used test the trained DNN on example image by reconstructing OD image based on inferred reference image.
-> Results will be save in ./results_predict/.
7. Example usage: Ubuntu -> Terminal -> Type: "./run.sh" and enter (Note. you have to give permission to run the shell script)
-> This will run train.py with configuration files contained in ./configs.
-> The results such as the visualization of inferred reference image and model weights will be saved in ./results/.
