import numpy as np
import os
import cv2

from keras.models import load_model

# 모델 불러오기
model = load_model('model.h5')

# 얼굴 인식
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

