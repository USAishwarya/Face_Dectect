# Face_Dectect 🔍
**CNN-based Face Detection** — training, evaluation, and real-time inference (webcam & batch images).

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Status](https://img.shields.io/badge/Status-Active-orange.svg)]

---

## Project overview
This repository contains a lightweight Convolutional Neural Network (CNN) based face detection pipeline with tools for:
- data preprocessing & augmentation  
- model training and checkpointing  
- evaluation (Precision / Recall / mAP)  
- inference (webcam, single image, batch folder)  
- saving cropped faces for downstream tasks

The project is designed to be easy to reproduce and extend for research or small production demos.

---

## Features
- CNN-based detector (trainable on custom datasets)  
- Real-time webcam inference with bounding boxes  
- Batch image processing and annotated-output saving  
- Option to save cropped face images for dataset expansion  
- Training logs, model checkpointing, and evaluation scripts

---

## Repo structure

Face_Dectect/
- data/ 
- models/ 
- outputs/ 
- src/
- train.py 
- detect.py 
- evaluate.py 
- model.py 
- dataset.py 
- utils.py
