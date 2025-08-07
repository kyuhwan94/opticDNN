# optDNN

**optDNN** is a deep learning-based tool for reconstructing atomic absorption images (or phase contrast images) with reduced fringe noise, by inferring the proper imaging beam profile at the position of the atoms.

This project is part of a research conducted at the [Quantum Gas Laboratory](https://qgl.snu.ac.kr/).

---

## 🚀 Features

- PyTorch-based training pipeline for image reconstruction.
- Pretraining and transfer learning support.
- Visualization of inferred reference (beam profile) images.
- Fringe removal without experimental reference images (single-shot absorption imaging is possible).
- Configurable training and evaluation pipeline.

---

## 📁 Repository Structure

- **configs/**: Example .json configuration files (pretraining/transfer modes)
- **results/**: Training outputs (inferred images, weights, logs)
- **results_predict/**: Results from prediction
- **src/**: Source code
  - `train.py`: Main training pipeline
  - `predict.py`: Inference script
  - `run.sh`: Shell script to run training

## ⚙️ Configuration Parameters (in `configs/default.json`)

| Parameter           | Description |
|---------------------|-------------|
| `ENABLE_PRETRAIN`   | Enable pretraining mode (True/False) |
| `ENABLE_REPLAY`     | Enable replay buffer for transfer learning |
| `ENABLE_TRANSFER`   | Enable transfer learning mode |
| `VAL_RATIO`         | Fraction of validation data (0–1) |
| `REPLAY_RATIO`      | Replay buffer size ratio (0–1) |
| `SAVE_PATH`         | Path to save outputs |
| `SAVE_WEIGHTS`      | File path to save model weights |
| `LOAD_WEIGHTS`      | Path to pretrained weights |
| `PATIENCE`          | Early stopping patience |
| `IMPROVEMENT_THRESHOLD` | Threshold to detect validation improvement |
| `FEATURES`, `NUM_GROUPS` | Model architecture options |
| `BATCH_SIZE`, `BATCH_MOMENTUM` | Training hyperparameters |
| `PRETRAIN_LR`, `TRANSFER_LR`, `WEIGHT_DECAY` | Learning rates and regularization |
| `CENTER_X`, `CENTER_Y`, `RADIUS` | Region-of-interest settings |
| `PRETRAIN_FOLDER_PATHS`, `EVAL_FOLDER_PATHS`, `TRANSFER_FOLDER_PATHS` | Input data folders |

---

## 🧪 Quickstart

```bash
# Give permission to run the training shell script
chmod +x ./src/run.sh

# Run training using default config
./src/run.sh
```

---

## 🔍 Prediction

To apply the trained model to new atomic absorption images:

```bash
python ./src/predict.py --config ./configs/default.json
```

- Prediction results will be saved in the ./results_predict/ directory.
- Make sure LOAD_WEIGHTS is set correctly in the config file to point to a trained model.

---

## 📌 Notes

- Tested on Linux with Python 3.10.12 and PyTorch 2.5.1
- Tested with Nvidia RTX 4090.
- GPU acceleration is highly recommended for training. 
