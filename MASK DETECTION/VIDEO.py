########  Importing necessary libraries
import cv2
import sys
from playsound import playsound
import json
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

########  Connecting to my watson studio to use the trained model
authenticator = IAMAuthenticator('{API KEY}')
#API KEY OD VISUAL RECOGNITION

visual_recognition = VisualRecognitionV3(
    version='{VERSION // DATE}',
    authenticator=authenticator)

visual_recognition.set_service_url('https://api.us-south.visual-recognition.watson.cloud.ibm.com/') 

########   Haarcascade for face detection
faceCascade = cv2.CascadeClassifier('C:/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

########   Taking live feed from the webcam
video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert to gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect if faces founf 
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
    )

    
    cv2.imwrite("test.jpg",frame)
    with open("test.jpg", 'rb') as images_file: 
        classes = visual_recognition.classify(images_file=images_file,threshold='0.6',classifier_ids='DefaultCustomModel_893593661').get_result()

    #print(json.dumps(classes, indent=2))    
    l=json.dumps(classes, indent=2)


    print(classes["images"][0]["classifiers"][0]["classes"][0]["class"])

        

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_img = frame[y:y+h,x:x+w]
        
        cv2.putText(frame, classes["images"][0]["classifiers"][0]["classes"][0]["class"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 2)
        
    if len(faces)==0:
        cv2.putText(frame, classes["images"][0]["classifiers"][0]["classes"][0]["class"], (int((frame.shape[0])/2), int((frame.shape[1])/2)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0), 2)
    cv2.imshow("img",frame)

    # Display the resulting frame

    if classes["images"][0]["classifiers"][0]["classes"][0]["class"] == "WITHOUT MASK IMAGE":
        playsound(r'C:\Users\hp\Downloads\beep-01a.mp3')
 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
















































'''
import numpy as np
import cv2

cat_cascade = cv2.CascadeClassifier('C:/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


SF=1.05  # try different values of scale factor like 1.05, 1.3, etc
N=3 # try different values of minimum neighbours like 3,4,5,6

def processImage(image_filename):
    # read the image
    img = cv2.imread(image_filename)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    
    
    for (x, y, w, h) in cats:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_img = img[y:y+h,x:x+w]
        cv2.imwrite("test.jpg",img)
        with open("test.jpg", 'rb') as images_file: 
            classes = visual_recognition.classify(images_file=images_file,threshold='0.6',classifier_ids='MaskDetection_736199640').get_result()

        #print(json.dumps(classes, indent=2))    
        l=json.dumps(classes, indent=2)


        print(classes["images"][0]["classifiers"][0]["classes"][0]["class"])

        cv2.putText(img, classes["images"][0]["classifiers"][0]["classes"][0]["class"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0), 2)   
    
    
    cv2.imshow('out',img)



processImage('H:/Face-Mask-Detection/test/test2.jpg')



@anupama
DefaultCustomModel_893593661
'''
