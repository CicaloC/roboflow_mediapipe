# Roboflow Mediapipe 

## Installation Guide
1. Clone this repository and open folder
''' git clone https://github.com/CicaloC/roboflow_mediapipe.git '''
''' cd roboflow_mediapipe

2. Create a Python3.7 Anaconda environment and activate it 

''' conda create -n roboflow_mediapipe_env python=3.7 '''
''' activate roboflow_mediapipe_env

3. Install Freemocap and Roboflow Dependencies

'''$ pip install -r requirements.txt '''
*Must be in the roboflow_mediapipe folder

## Process Videos

1. Go to line 618 of roboflow_mediapipe.py and change yolo_path to the 
folder path that you cloned the repo to. Change sourceFolder to folder
of input videos and outSourceFolder to where you want output videos saved

2. Run roboflow_mediapipe.py