import cv2, glob
import dlib
import matplotlib.pyplot as plt
import numpy as np
global frame

capture = cv2.VideoCapture(0)  #웹캠을 객체로 만듭니다.
capture.set(3, 640)  #픽셀길이 가로 640
capture.set(4, 480)  #픽셀길이 세로 480
print("작동")
while True:  #'q'키를 누를 때까지 반복
    ret, frame = capture.read()  #카메라로부터 영상 하나 읽어옵니다.
    cv2.imshow('frame', frame)  # 영상을 window 에 표시합니다.
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    key_2 = cv2.waitKey(60)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        img_name = "../GraduationProject/img/opencv_frame.jpg"
        cv2.imwrite(img_name, frame)  # 영상에서 캡쳐한 이미지를 저장합니다.
        break
    elif key_2 ==ord('s'):
        cv2.imwrite('./img/test.jpg',frame)
        break
    
#age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 19)','(20, 25)','(26,32)','(33,37)','(38, 43)','(48, 53)','(60, 100)']



age_net = cv2.dnn.readNetFromCaffe(
          'model/age.prototxt', 
          'model/age.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe(
          'model/gender.prototxt',
          'model/gender.caffemodel')


img_list = glob.glob('img/*.jpg')
cnt = 0

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


for img_path in img_list:
  cnt+=1
  img = cv2.imread(img_path)
  faces = detector.detectMultiScale(img,1.3,5)

  try:
    x,y,w,h = faces[0]
    
    
    detected_face = img[int(y):int(y+h) , int(x):int(x+w)]
    detected_face = cv2.resize(detected_face,(224,224))
    detected_face_blob = cv2.dnn.blobFromImage(detected_face)
    
    age_net.setInput(detected_face_blob)
    age_result = age_net.forward()
    
    idx = np.array([i for i in range(1,102)])
    age = round(np.sum(age_result[0] *idx))
    
    
    gender_net.setInput(detected_face_blob)
    gender_result = gender_net.forward()
    
    if np.argmax(gender_result[0]) == 0:
        gender = "Female"
    else:
        gender = 'Male'

    
      
    print(gender,age)
    x1 = int(x)
    y1 = int(y)
    x2 = int(x+w)
    y2 = int(y+h)
    # visualize
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
    overlay_text = '%s %s' % (gender, age)
    cv2.putText(img, overlay_text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=1, color=(0,0,0), thickness=10)
    cv2.putText(img, overlay_text, org=(x1, y1),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

    cv2.imshow('img', img)
    cv2.imwrite('./result/result_%s.jpg'%cnt, img)
  except:
    cv2.imshow('img', img)
    cv2.imwrite('./result/result_%s.jpg'%cnt, img)
    print("Failed to Find Face")

  key = cv2.waitKey(0) & 0xFF
  if key == ord('q'):
    break

# def Webcam():
#     global frame
#     capture = cv2.VideoCapture(0)  #웹캠을 객체로 만듭니다.
#     capture.set(3, 640)  #픽셀길이 가로 640
#     capture.set(4, 480)  #픽셀길이 세로 480
#
#     while True:  #'q'키를 누를 때까지 반복
#         ret, frame = capture.read()  #카메라로부터 영상 하나 읽어옵니다.
#         cv2.imshow('frame', frame)  # 영상을 window 에 표시합니다.
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             img_name = "C:/Users/USER/PycharmProjects/GraduationProject/opencv_frame.jpg"
#             cv2.imwrite(img_name, frame)  # 영상에서 캡쳐한 이미지를 저장합니다.
#             break


    # global frame
    #
    # capture_counter = 1
    # start_time = time.time()

    # while True:  #'q'키를 누를 때까지 반복
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #       img_name = "C:/Users/USER/PycharmProjects/GraduationProject/opencv_frame.jpg"
    #       cv2.imwrite(img_name, frame)  # 영상에서 캡쳐한 이미지를 저장합니다.
    #       break
    #     # if time.time() - start_time >= views.leng:  #<---- (광고시간1)초 뒤에 캡쳐합니다.
    #     #     start_time = time.time()
    #     #     img_name = "C:/Users/USER/PycharmProjects/GraduationProject/opencv_frame.jpg"
    #     #     cv.imwrite(img_name, frame)  #영상에서 캡쳐한 이미지를 저장합니다.



"""
import sys
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('fail')

while True:
    _, frame = cap.read()
    cv2.imshow('123', frame)
    key = cv2.waitKey(60)
    if key == 27:
        break
    elif key ==ord('s'):
        ret,frame = cap.read()
        
        cv2.imwrite('./img/test.jpg',frame)
        break
"""

