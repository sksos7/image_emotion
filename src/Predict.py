import os
import sys

import cv2
import numpy as np

from PIL import ImageFont, ImageDraw, Image

from keras.models import load_model
from keras.preprocessing import image

# 모델 불러오기
model = load_model('src\model.h5')

# 얼굴 인식
eye_cascade = cv2.CascadeClassifier('src/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')

# 예측할 이미지 파일 경로
input_argv = sys.argv[1]
img_file_path = 'Predict_img/' + input_argv

# 이미지 불러오기 한글 경로 깨짐
img_array = np.fromfile(img_file_path, np.uint8)
img_decode = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# 이미지의 사이즈 변환
img_resize = cv2.resize(img_decode, dsize=(640,480))

# 그레이 스케일 변환
img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY) 

# cascade 얼굴 탐지 알고리즘
face_results = face_cascade.detectMultiScale(img_gray)

# 감지된 영역 박스
for face_box in face_results:
    x, y, w, h = face_box
    # img_resize 이미지에 빨간 박스 그리기
    cv2.rectangle(img_resize , (x,y), (x+w, y+h), (0,0,255), thickness=2)

    # 빨간 박스 부분의 이미지만 따로 저장
    img_cropped = img_resize[y:y + h, x:x + w]
    gray_cropped = img_gray[y:y + h, x:x + w]

    # 인식된 부분에서 눈이 2개 이상 인식하면
    eye_results = eye_cascade.detectMultiScale(img_cropped)
    if len(eye_results) >= 2:
        # 감지된 영역의 이미즈를 모델 입력값에 맞는 사이즈로 조정
        cropped_resize = cv2.resize(gray_cropped,dsize=(64, 64))

input_img = np.array([cropped_resize])

# 예측
result = model.predict(input_img)
predict =  np.argmax(result, axis=-1)

emo = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']


# cv2 에 한글이 보이도록
img = Image.fromarray(img_resize) #img배열을 PIL이 처리가능하게 변환
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("fonts/gulim.ttc", 21)

res_text = '결과 : ' + emo[predict[0]]
for i in range(0, 6):
    res_text += '\n' + emo[i]+" : "+ format(result[0][i], 'f')

for i, line in enumerate(res_text.split('\n')):
    y = y + 20
    draw.text((x+h, y), line, font=font, fill=(0,0,255)) #text를 출력

#다시 OpenCV가 처리가능하게 np 배열로 변환
img = np.array(img)

# 화면 출력
cv2.imshow('Face Emotion',img)
        
cv2.waitKey(10000)
cv2.destroyAllWindows()