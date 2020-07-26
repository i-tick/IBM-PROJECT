import cv2
import sys
from playsound import playsound
import json
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator



########  Connecting to my watson studio to use the trained model
authenticator = IAMAuthenticator('{API KEY}')
#API KEY OF VISUAL RECOGNITION

visual_recognition = VisualRecognitionV3(
    version='{VERSION // DATE}',
    authenticator=authenticator)

visual_recognition.set_service_url('https://api.us-south.visual-recognition.watson.cloud.ibm.com/') 

########   Haarcascade for face detection
#faceCascade = cv2.CascadeClassifier('C:/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

########   Taking live feed from the webcam

with open("H:\mask-detection-master\data\mask\WIN_20200719_00_43_25_Pro.jpg", 'rb') as images_file: 
    classes = visual_recognition.classify(images_file=images_file,threshold='0.6',classifier_ids='DefaultCustomModel_893593661').get_result()

#print(json.dumps(classes, indent=2))    
l=json.dumps(classes, indent=2)
print(classes["images"][0]["classifiers"][0]["classes"][0]["class"])
img = cv2.imread('H:\mask-detection-master\data\mask\WIN_20200719_00_43_25_Pro.jpg')
cv2.putText(img, classes["images"][0]["classifiers"][0]["classes"][0]["class"], (int((img.shape[0])/2), int((img.shape[1])/2)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0), 2)
cv2.imshow("img",img)


#H:/mask-detection-master/data/no-mask/WIN_20200719_00_45_02_Pro.jpg
