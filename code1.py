import cv2
import pyttsx3
import os
engine = pyttsx3.init()

thres = 0.45  #Threshold to detect object
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
 
classNames= []
classFile = (r'C:/Users/bvdis/blindassist/blindassist/coco.names')
with open(classFile,'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = (r'C:/Users/bvdis/blindassist/blindassist/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = (r'C:/Users/bvdis/blindassist/blindassist/frozen_inference_graph.pb')

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
 
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)
 
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            if classId < 91 :
                print (classNames[classId-1])
                
                print('audio begin')
                s = classNames[classId-1]
                #os.system(s)
                engine.say(s)
                engine.runAndWait()
 
    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
