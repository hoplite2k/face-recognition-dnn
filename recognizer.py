import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import pickle
import os
import time

#paths of all pre-trained models
modelpath = './pre_trained_models/model/res10_300x300_ssd_iter_140000.caffemodel'
prototxtpath = './pre_trained_models/prototxt/deploy.prototxt.txt'
embedderpath = './pre_trained_models/embedder/openface_nn4.small2.v1.t7'

#load detector
print("[INFO] loading face detector...")
detector = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)

#load embedder
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedderpath)

#load recognizer
recognizer = pickle.loads(open('./recognizer/recognizer.pickle','rb').read())
le = pickle.loads(open('./label_encoder/le.pickle','rb').read())

#start stream
print("[INFO] starting video stream...")
cam = VideoStream(0).start()
time.sleep(1)

fps = FPS().start()

#one frame at a time
while(True):
    #get frame
    frame = cam.read()
    frame = cv2.flip(frame, +1)
    frame = imutils.resize(frame, width=600)
    (h,w) = frame.shape[:2]

    #create blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    #feed blob to the detector
    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        #find face portion
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7]*np.array([w,h,w,h])
            (x,y,ex,ey) = box.astype("int")
            #get face portion
            face = frame[y:ey, x:ex]
            (fh,fw) = face.shape[:2]
            if fh<20 or fw<20:
                continue

            #create face blob
            faceblob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

            #feed face blob to embedder and get its corresponding embeddings
            embedder.setInput(faceblob)
            embedding = embedder.forward()

            #get predictions from recognizer and keep only the highest prediction
            preds = recognizer.predict_proba(embedding)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            #bounding box and name
            text = "{}".format(name)
            #printing probability aslo
            #text = "{}: {:.2f}%".format(name, proba * 100)
            y_ = y - 10 if y - 10 > 10 else y + 10
            cv2.rectangle(frame, (x,y), (ex,ey), (0,0,255), 2)
            cv2.putText(frame, text, (x,y_), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #diplay screen
    fps.update()
    cv2.imshow("cam",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
cam.stop()
