# atomDNN
This registry contains a python script for training a deep neural network, which can be used to reconstruct atomic absorption signals.

This repository is maintained by [Quantum Gas Laboratory](https://qgl.snu.ac.kr/).

## Items

1. **src**: Contains source code for training

2. **configs**: Contains an example configuration that can be used for training.
 
   * The example configuration is set in pretraining mode. 



## Pipeline
5. ./src/run.sh: Shell script used to execute train.py
6. ./src/train.py: Pipeline of the DNN (PyTorch)
   * Train via back propagation (ADAMW)
   * Visualization of inferred reference image
8. ./src/predict.py: Python code that can be used test the trained DNN on example image. $\rightarrow$ Results will be save in ./results_predict/.

## Example usage
Terminal (in our case the OS was Linux) $\rightarrow$ Type: "./run.sh" and enter (Note. you have to give permission to run the shell script)
$\rightarrow$ This will run train.py with configuration files contained in ./configs.
$\rightarrow$ The results such as the visualization of inferred reference image and model weights will be saved in ./results/.
