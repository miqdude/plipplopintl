import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import signal
import multiprocessing
import array
from cv2 import dnn

caffeModel = "MobileNetSSD_deploy.caffemodel"
caffeProto = "MobileNetSSD_deploy.prototxt.txt"

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

classNames = ('background',
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')

net  = cv2.dnn.readNetFromCaffe(caffeProto, caffeModel)

"""
Net Caffe
"""

def caffe(frame):
    blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), meanVal)

    net.setInput(blob)
    detections = net.forward()

    cols = frame.shape[1]
    rows = frame.shape[0]

    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))

    y1 = (rows - cropSize[1]) / 2
    y2 = y1 + cropSize[1]
    x1 = (cols - cropSize[0]) / 2
    x2 = x1 + cropSize[0]
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    frame = frame[y1:y2,x1:x2]

    cols = frame.shape[1]
    rows = frame.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > .2:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame


"""
YOLOV3
"""

# read class names from text file
classes = None
with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

netyolo = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

def yolo(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)

    netyolo.setInput(blob)

    outs = netyolo.forward(get_output_layers(netyolo))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # conf_threshold = 0.5
    # nms_threshold = 0.4

    Width = frame.shape[1]
    Height = frame.shape[0]

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

    car = 0
    bus = 0
    motorcycle = 0
    truck = 0
    others = 0

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        if str(classes[class_ids[i]]) == "car" :
            car+=1
        elif str(classes[class_ids[i]]) == "bus":
            bus+=1
        elif str(classes[class_ids[i]]) == "motorcycle":
            motorcycle+=1
        elif str(classes[class_ids[i]]) == "truck":
            truck+=1
        else:
            others+=1        
    
    res = []
    res.append(car)
    res.append(bus)
    res.append(motorcycle)
    res.append(truck)
    res.append(others)

    ret = res[0]*1.5 + res[1]*2 + res[2]*1 + res[3]*2

    tick = 0

    if ret < 12:                    # agak sepi
        tick = 10
    elif ret >= 12 and ret < 20:    # berisi
        tick = 30
    elif ret >= 20:                 # padat
        tick = 42

    return tick,frame

print('initialize server')

def signal_interrupt(sig, frame):
    print('exit')
    sys.exit(0)

def cam1process():

    HOST=''
    PORT=8989

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn,addr=s.accept()
    signal.signal(signal.SIGINT,signal_interrupt)
    
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))


    PI = '192.168.137.188'
    PIPORT = 8787

    
    while True:
        while len(data) < payload_size:
            print("Recv2: {}".format(len(data)))
            data += conn.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # image = caffe(frame)
        res,frame = yolo(frame)
        print("the result are {}".format(res))

        cv2.imshow('ImageWindow',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        #sent.close()
        sent = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sent.connect((PI,PIPORT))

        sent_data = struct.pack('B',res)
        #sent_data_size = len(sent_data)
        # ini tolong diisi timer baru hasil itung2an gan
        print("the sent data {}".format(sent_data))
        sent.sendall(sent_data)
        sent.close()

        signal.signal(signal.SIGINT,signal_interrupt)

    cv2.destroyAllWindows()

jobs = []
if __name__ == "__main__":
    pros1 = multiprocessing.Process(target = cam1process)
    jobs.append(pros1)
    pros1.start()