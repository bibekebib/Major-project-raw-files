
# from google.colab import drive
# drive.mount('/content/drive')

dataset_path = '/content/drive/MyDrive/dataset'

from glob import glob
import pandas as pd
import cv2 as cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
pip install mediapipe


import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192)
def landmark_det(image):
  with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    a = []
    image_height, image_width, _ = image.shape
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i in range(33):
      a.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].x ,results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].y ,results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].z))
    b = []

    for i in a:
      for x in i:
        b.append(x)
    df = pd.DataFrame(b).T
    df = df.to_numpy()
    return df


import os


def listframe(path):
  local = []
  video = cv2.VideoCapture(path)
  frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  f_img = 0
  count = 0
  while (count<frame_num): 
      try:
          success, cap = video.read()
          if count % (int(frame_num / 256)) == 0 and (f_img<256):
              df  = landmark_det(cap)
              local.append(df)
              f_img =f_img+1
              count = count+1
          else:
              count = count+1
          
      except:
        count = count+1
        print(path)
        print(count)
  print(f'done dong doing {path}')

  return local


data = np.array(listframe('file_path'))
globals = []
for i in range(len(data)):
    for j in range(len(data[i])):
        globals.append(data[i][j])

total = []
for i in range(len(globals)):
  total.append((globals[i][0][0],globals[i][1]))

df = pd.DataFrame(total)

df_test = pd.concat([pd.DataFrame(df[0].values.tolist()), df[1]], axis=1)



