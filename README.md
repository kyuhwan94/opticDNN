# optDNN

**optDNN** is a deep learning-based tool for reconstructing atomic absorption images with reduced fringe noise, by inferring the proper imaging beam profile at the position of the atoms.

This project is part of a research conducted at the [Quantum Gas Laboratory](https://qgl.snu.ac.kr/).

---

## 🚀 Features

- PyTorch-based training pipeline for image reconstruction.
- Pretraining and transfer learning support.
- Visualization of inferred reference (beam profile) images.
- Fringe removal without experimental reference images.
- Configurable training and evaluation pipeline.

---

## 📁 Repository Structure



## Items

### _src_: Contains source code for training

#### train.py

* Contains both the pipeline of the DNN and training protocol.
* Also, a visualization of the inferred reference image shows whether we are on the right track.

----

### _configs_: Contains an example configuration that can be used for training.
 
   * The example configuration is set in pretraining mode.

#### Contents of configuration

   * ENABLE_PRETRAIN: Toggle switch to enable pretrain (Set to true to enable pretrain. Set to false if you plan to load pretrained weights)
   * ENABLE_REPLAY: Toggle switch to enable replay while transfer learning
   * ENABLE_TRANSFER: Toggle switch to enable transfer learning
   * VAL_RATIO: The portion of how much you are going to reserve for validation (values between 0 and 1)
   * REPLAY_RATIO: The portion of how much you are going to use for replay (values between 0 and 1)
   * SAVE_PATH
   * SAVE_WEIGHTS
   * LOAD_WEIGHTS
   * PATIENCE
   * IMPROVEMENT_THRESHOLD
   * FEATURES
   * NUM_GROUPS
   * BATCH_SIZE
   * BATCH_MOMENTUM
   * PRETRAIN_LR
   * TRANSFER_LR
   * WEIGHT_DECAY
   * CENTER_X
   * CENTER_Y
   * RADIUS
   * PRETRAIN_FOLDER_PATHS
   * EVAL_FOLDER_PATHS
   * TRANSFER_FOLDER_PATHS

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
