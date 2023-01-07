from turtle import color
import cv2
from cv2 import CV_32F 
import numpy as np 

cap = cv2.VideoCapture("akıncı.mp4")
# cap = cv2.VideoCapture(0)

while True : 
    ret, frame = cap.read()
    
    frame = cv2.flip(frame,1)
    # frame = cv2.resize(frame,(960,720))

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True , crop= False)

    labels = ["IHA"]

    colors = ["0,0,255","255,0,0","255,0,255"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))

    model = cv2.dnn.readNetFromDarknet("yolov4_iha.cfg", "yolov4_iha_best.weights")

    layers = model.getLayerNames()
    output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)

    detection_layers = model.forward(output_layer)


    ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in detection_layers : 
        for object_detection in detection_layer : 

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.20 : 

                label = labels[predicted_id] 

                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")


                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))



                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])






    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

        
    for max_id in max_ids:
        
        max_class_id = max_id
        box = boxes_list[max_class_id]
        
        start_x = box[0] 
        start_y = box[1] 
        box_width = box[2] 
        box_height = box[3] 
         
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
      
                
        end_x = start_x + box_width
        end_y = start_y + box_height
                
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
                
                
        label = "{}: {:.2f}%".format(label, confidence*100)
        #print("predicted object {}".format(label))
         
                
        cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),box_color,2)
        cv2.rectangle(frame, (start_x-1,start_y),(end_x+1,start_y-30),box_color,-1)
        cv2.putText(frame,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    
    if cv2.waitKey(1) & ord("q") ==27:
        break
    
    cv2.imshow("UAV DETECTION",frame)








