# atomDNN
This registry contains the deep neural network architecture and training protocol which can be used to reconstruct atomic absorption signals.

We provide the python script used for training the network.

This repository is [Quantum Gas Laboratory] (https://qgl.snu.ac.kr/).

## Items
1. ./configs: Contains an example configuration that can used for training.
> * The example configuration is set in pretraining mode. 
3. ./src: Contains source code for training

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
