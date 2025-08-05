import cv2 as cv
import numpy as np
import csv
from sort.sort import Sort
from ultralytics import YOLO
from util import *
#Model
car_detector = YOLO('yolov8n.pt')
plate_detector = YOLO(r'license_plate_detector.pt')


#Camera 
capture = cv.VideoCapture(r'sort\data\videos\morocco road.mp4')
fps = capture.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps) 


#Loging
csv_file = open('detections.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'x1', 'y1', 'x2', 'y2', 'track_id'])

#Tracker
tracker = Sort()

vehicles = [2, 3, 5, 7]
#vehicles = [i for i in range(1000)]

framenum = 0
while True :
    
    isTrue, frame = capture.read()
    if not isTrue:
        break
    
    frame = cv.resize(frame,( 1200 , 1200* frame.shape[0]//frame.shape[1]), interpolation=cv.INTER_CUBIC)
    framenum +=1    
    
    #vehicle detections
    result = car_detector(frame)[0]
    detections = []
    
    for data in result.boxes.data:
        x1, y1, x2, y2, conf, cls_id = data
        conf = conf.item()
       
        
        if (cls_id in vehicles) and (conf > 0.5) :
            detections.append([x1, y1, x2, y2,conf])
    detections = np.array(detections) if len(detections)>0 else np.empty((0,5))
                        
    tracks = tracker.update(detections)
    
    plate_result = plate_detector(frame)[0]   
    
    track_id_to_plate = plate_to_track_id(plate_result.boxes.data, tracks)
    print(track_id_to_plate)
    
   
    
    for track_id in track_id_to_plate:
        x1_pl, y1_pl, x2_pl, y2_pl = map(int,track_id_to_plate[track_id])
        plate_cropped = frame[y1_pl:y2_pl, x1_pl:x2_pl]
        
        if plate_cropped.size > 0:
            plate_resized = cv.resize(plate_cropped, (200, 60))  # or any size
            cv.imshow(f'plate_{track_id}', plate_resized)

        
        
        cv.rectangle(frame,(int(x1_pl), int(y1_pl)), (int(x2_pl), int(y2_pl)), (0,255,0), 2)
        cv.putText(frame,f'ID: {track_id}',(int(x1_pl), int(y1_pl)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        
     
    for track in tracks :
        x1, y1, x2, y2, track_id = track
        csv_writer.writerow([framenum, x1, y1, x2, y2, track_id])
        cv.rectangle(frame,(int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv.putText(frame,f'ID: {track_id}',(int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        
    
    if cv.waitKey(1) == ord('q'):
        break
    
    
    
    cv.imshow('Road', frame)
    
csv_file.close()
capture.release()
cv.destroyAllWindows()
