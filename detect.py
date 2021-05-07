from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 
import sys
import numpy as np


if len(sys.argv)!=2:
    print("Correct usage:python detect.py filename")
    sys.exit(0)
else:
    img=str(sys.argv[1])
    test_image=cv2.imread(img)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model =load_model('Emotion_little_vgg.h5')

    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
    # Grab single frame from video
    labels=[]
    gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            # rect,face,image = face_detector(frame)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            # make prediction on ROI, then lookup the class

            preds = model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x,y)
            out = cv2.putText(test_image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        else:
            out = cv2.putText(test_image,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    out=cv2.resize(out,(760,760))
    cv2.imshow('Emotion Detector',out)
    cv2.waitKey(0)
