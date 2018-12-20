import socket
import numpy as np
import cv2
import argparse
import sys
import struct
import signal
import pickle

classes = None
with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# get layer jenis kendaraan
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


# load warna kotak2nya
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# load trained weight dan configurasi
netyolo = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

def yolo(frame):
    # frame = cv2.imread(filename)
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

    ret = res[0]*1.5 + res[1]*2 + res[3]*1.75 + res[2]*1 + res[4]*.5

    tick = 0

    if ret < 12:                    # agak sepi
        tick = 10
    elif ret >= 12 and ret < 20:    # berisi
        tick = 30
    elif ret >= 20:                 # padat
        tick = 42

    # cv2.imshow("yolo_",frame)  
    return tick, frame

"""
Interupt signal CTRL+C
"""
def signal_interrupt(sig, frame):
    print('exit')
    sys.exit(0)

"""
function return data ke RaspberryPi
"""
def return_data(res,addr):
    sent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("addr {} {}".format(addr[0],addr[1]))
    sent.settimeout(5)
    try:
        sent.connect((addr[0],8787))
        sent_data = struct.pack('B',res)
        # sent_data_size = len(sent_data)
        # ini tolong diisi timer baru hasil itung2an gan
        # print("the sent data {}".format(sent_data))
        sent.sendall(sent_data) # Sent data
    except socket.error as e:
        print(e)
    
    sent.close()        
    return

"""
Main server function untuk terima data gambar
"""
def serve(conn,addr):
    print("serving")

    data = b""
    payload_size = struct.calcsize(">L")

    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)

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

        print("yolo..")
        res, frame= yolo(frame)
        print("result is {}".format(res))
        return_data(res,addr)

        cv2.imshow("wow data",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)


"""
Function server menunggu request koneksi dari client
"""
def server_start(port):
    print("starting server")

    try:
        sok = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sok.bind(('0.0.0.0',port))
    except socket.error as e:
        print("error couldn't start server {}".format(e))
        sys.exit(0)

    sok.settimeout(5)
    sok.listen(0)
    print("listening")

    while True:
        try:
            signal.signal(signal.SIGINT,signal_interrupt)
            conn, addr = sok.accept()
        except socket.error as e:
            print("no incoming connection, looping")
            continue
        else:
            serve(conn,addr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('ip',help='ip address')
    parser.add_argument('socket', type=int,help='socket number (must above 1024)')
    
    # parser.add_argument('dest_ip',help='destinantion ip address')
    # parser.add_argument('dest_socket', type=int,help='destination socket number')

    args = parser.parse_args()
    # print("{} {}".format(args.ip, args.socket))

    server_start(args.socket)