# Roboflow Mediapipe 

## Installation Guide
1. Clone this repository and open folder
```
git clone https://github.com/CicaloC/roboflow_mediapipe.git
cd roboflow_mediapipe
```
2. Create a Python3.7 Anaconda environment and activate it 
```
conda create -n roboflow_mediapipe_env python=3.7
activate roboflow_mediapipe_env
```
3. Install Freemocap and Roboflow Dependencies
```
$ pip install -r requirements.txt
```
*Must be in the roboflow_mediapipe folder*
If you want to run on GPU
```
pip uninstall torch
pip uninstall torchvision
pip install torchvision --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```
4. Download yolo model

Go to: https://drive.google.com/drive/u/0/folders/1scSTZVBw07IjnvICwQYiMMtqKxIPD8uq
Download either yolov5s.pt or yolov5x.pt and place the file in the models folder.
5s is a smaller network and will run faster but is less accurate than 5x. 5s is the
default weights, if using 5x, change line 644 in roboflow_mediapipe.py

In my experience mediapipe does not come downloaded with the model necessary to 
set model_complexity = 2. If that is the case than you can go to 
https://drive.google.com/drive/folders/15R-b6iP6jSt3HHKeJNhfvalhIJIpYeqO?usp=sharing
download that file and past it in: 
C:\Users\yourpath\anaconda3\envs\roboflow_mediapipe_env\Lib\site-packages\mediapipe\modules\pose_landmark


## Process Videos

1. Go to line 618 of roboflow_mediapipe.py and change yolo_path to the 
folder path that you cloned the repo to. Change sourceFolder to folder
of input videos and outSourceFolder to where you want output videos saved

2. Run roboflow_mediapipe.py
2. Run roboflow_mediapipe.py
