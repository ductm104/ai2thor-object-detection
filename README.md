# AI2Thor - Object detection using Yolo and RetinaNet

This is final project of final semester in robotic - INT3409-21

## Overview
- Download [Pretrained model](https://drive.google.com/drive/folders/1oKgneR-6lbXfJXuVyPSHhmMsjxXfyApH?usp=sharing) and put in `./models/`
- Run application: `python ./src/main.py`
## Installation

1. Clone this repository.
- `cd ./ai2thor-object-detection`
2. Install Python dependencies:  
- `pip install -r requirements.txt`
3. Clone Keras-RetinaNet repository under `src` folder:  
- `cd ./src`  
- `git clone https://github.com/fizyr/keras-retinanet keras_retinanet`  
- `cd ./keras_retinanet`  
- `python setup.py build_ext --inplace`