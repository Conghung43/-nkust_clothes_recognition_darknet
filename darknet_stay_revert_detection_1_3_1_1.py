from ctypes import *
import random
import os
import cv2
import time
import darknet
from queue import Queue
import threading
import numpy as np
import sys
import requests
from io import BufferedReader, BytesIO
from datetime import datetime
import json
import socket
import logging
import struct
import pickle
import zlib
import select
import configparser
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from threading import Lock
import re
import base64
import json
import requests
from pythonping import ping
import copy 
import logging.handlers

lock = Lock()

lock_func = False

bug_check = False

resp_time = " "

array_queue = {}

def send_image(image, extension, time, status, plate_number, vehicle_type, vehicle_color, zone_name, lane_name):
    try:
        # my_logger.debug(str(datetime.now()) + ' START send_image')
        print(str(datetime.now()) + ' START send_image ',extension, time, status, plate_number, vehicle_type, vehicle_color, zone_name, lane_name)
        image = cv2.resize(image, (960,480), interpolation = cv2.INTER_AREA)
        retval, buff = cv2.imencode(extension, image)
        jpg_as_text = base64.b64encode(buff)
        headers = {"api-key": "yt27h1j9b4"}
        myobj = {
                    "occurDatetimeMillis" :  int(time),
                    "violationReason" : status,
                    "base64picture": jpg_as_text.decode("utf-8"),
                    "plateNumber": plate_number,
                    "isViolation": True,
                    "vehicleType": vehicle_type,
                    "vehicleColor": vehicle_color,
                    "zoneName": zone_name,
                    "laneName": lane_name}

        myobj = json.dumps(myobj)
        response = requests.post('http://140.127.62.119/api/v1/violation/insert-to-redis', data = myobj, headers=headers)
        # my_logger.debug(str(datetime.now()) + ' END send_image ' + str(response.status_code))
        print(str(datetime.now()) + ' END send_image ' + str(response.status_code))
    except Exception as e:
        print(str(datetime.now()) + " Unexpected error in send_image: "+ str(e))
        # my_logger.debug(str(datetime.now()) + " Unexpected error in send_image: "+ str(e))


def polygon(label,image):

    for conf in poly_config[label]:
        temp_array.append(poly_config[label][conf])
    print(type(poly_config[label][conf]))

    for i in range(0,len(temp_array),2):
        coordinate_array.append((int(float(temp_array[i])*image.shape[1]),int(float(temp_array[i+1])*image.shape[0])))

    polygon = Polygon(coordinate_array)

    point = Point(i, j)
    if not polygon.contains(point):
        return True
    else:
        return False

class main_Thread (threading.Thread):
    
    def __init__(self, name, lane_name, rtsp, threadID):
        threading.Thread.__init__(self)
        self.name = name
        self.lane_name = lane_name
        self.rtsp = rtsp
        self.threadID = threadID
    def run(self):
        print("Starting " + self.name)
        collect_frame_call_yolo(self.name, self.lane_name, self.rtsp, self.threadID)
        print("Exiting " + self.name)

class sub_Thread (threading.Thread):
    
    def __init__(self, name, lane_name, threadID, fps):
        threading.Thread.__init__(self)
        self.name = name
        self.lane_name = lane_name
        self.fps = fps
        self.threadID = threadID
    def run(self):
        print("Starting yolo " + self.name)
        YOLO(self.name, self.lane_name, self.threadID, self.fps)
        print("Exiting yolo" + self.name)

class image_contener:
    def __init__(self, name, image):
        self.name = name
        self.image = image

def find_nearest_point(cX, cY, list_tracking, vehicle_size):
    distance = 0
    index = -1
    count = 0
    for k in range(0,len(list_tracking)):

        if len(list_tracking[k]) > 0:
            temp = np.sqrt(np.power(list_tracking[k][-1][0] - cX,2) + np.power(list_tracking[k][-1][1] - cY,2))
            count += 1
            if count == 1:
                distance = temp
                index = k
            else:
                if distance >= temp:
                    distance = temp
                    index = k
        
    if distance > 80:#vehicle_size*2:
        index = -1
    return index

def add_new_array(list_tracking, list_tracking_lenth, list_tracking_flag, total_object):
    for i in range(total_object):
        if len(list_tracking) < total_object:
            list_tracking.append(np.zeros(shape=(0,2)))
            list_tracking_lenth.append(0)
            list_tracking_flag.append(False)
    return list_tracking, list_tracking_lenth, list_tracking_flag

def get_object_coordinate(frame, coordinate):
    width, height = frame.shape[:2]
    top = coordinate[0]*width #ymin
    left = coordinate[1]*height #xmin
    bottom = coordinate[2]*width #ymax
    right = coordinate[3]*height #xmax

    cX = int((left + right) / 2.0)
    cY = int((top + bottom) / 2.0)
#     cY = int(bottom)
    return cX,cY

def clustering(name, lane_name, list_tracking, list_tracking_lenth, list_tracking_flag, pt1, pt2, object_color, vehicle_size, frame, threadID, oldest_file, fps):
    #Find Chanel and add into
    global bug_check
    my_logger.debug(str(datetime.now()) + ' START clustering')
    cX = int((pt1[0] + pt2[0])/2)
    cY = int(pt2[1])
    chanel = find_nearest_point( cX, cY, list_tracking, vehicle_size) 

    img_shape = frame.shape

    coutour_frame = int(re.findall(r"\d+", oldest_file)[1])

    # Avoid wrong revert detection
    if len(list_tracking[chanel]) > 1:
    	m = cX
    	n = list_tracking[chanel][-1][0]
    	x = img_shape[1]*float(ch_config[threadID]['seperate_line'])
    	is_dif_side = m*n - x*(m + n) < -x*x
    	if is_dif_side:
    		chanel = -1

    if chanel >= 0: 

        top_border = float(ch_config[threadID]['top'])* img_shape[0]
        button_border = float(ch_config[threadID]['button'])* img_shape[0]
        is_enable_left = eval(ch_config[threadID]['is_enable_left'])
        is_enable_right = eval(ch_config[threadID]['is_enable_right'])
        is_enable_stop = eval(ch_config[threadID]['is_enable_stop'])
        is_enable_revert = eval(ch_config[threadID]['is_enable_revert'])
        seperate_line = float(ch_config[threadID]['seperate_line'])*img_shape[1]

        if len(list_tracking[chanel]) >= 0 and (cY > button_border or cY < top_border):
            list_tracking[chanel] = np.zeros(shape=(0,2))
            list_tracking_lenth[chanel] = 0
            list_tracking_flag[chanel] = False
            return list_tracking, list_tracking_lenth, list_tracking_flag

        list_tracking[chanel] = np.append(list_tracking[chanel],[[cX,cY]],axis = 0)
        list_tracking_lenth[chanel] = list_tracking_lenth[chanel] + 1 + coutour_frame
        # if is_add_list_tracking_lenth:
        #     list_tracking_lenth[chanel] += int(ch_config['countour_time']['countour_second'])
        # list_of_color[chanel] = object_color
        # print(list_tracking_lenth[chanel])
        
        number_of_step = int(ch_config['reverse_detection']['number_of_step'])
        compare_index = -1 - number_of_step
        distance_index = int(ch_config['reverse_detection']['distance'])

        # Stay detection code here
        
        if is_enable_stop and list_tracking_lenth[chanel] > int(fps) * int(ch_config['stay_detection']['stay_second']):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.rectangle(frame, pt1, pt2, [0,0,255], 2)
            if is_enable_left and cX < seperate_line:
                cv2.imwrite(r'image_Log\stay_detection_left_%d_%s.jpg' %(round(datetime.now().timestamp()*1000),threadID), frame)
                thread = threading.Thread(target=send_image, args = (frame, '.jpg', oldest_file[:13], 'IllegalParking', [], '', 'None', name, lane_name)).start()
            elif is_enable_right and cX > seperate_line:
                cv2.imwrite(r'image_Log\stay_detection_right_%d_%s.jpg' %(round(datetime.now().timestamp()*1000),threadID), frame)
                thread = threading.Thread(target=send_image, args = (frame, '.jpg', oldest_file[:13], 'IllegalParking', [], '', 'None', name, lane_name)).start()
            list_tracking_lenth[chanel] = 0

        # reverse detection code here
        # reverse_flag_header = False
        reverse_flag = False

        # # check header array
        # if len(list_tracking[chanel]) == 9 :
        #     if cX < seperate_line:
        #         if list_tracking[chanel][0][1] > list_tracking[chanel][8][1] :
        #             reverse_flag_header = True
        #     else:
        #         if list_tracking[chanel][0][1] < list_tracking[chanel][8][1] :
        #             reverse_flag_header = True

        # check array
        if is_enable_revert and len(list_tracking[chanel]) > 15 and abs(list_tracking[chanel][-1][1] - list_tracking[chanel][-15][1]) > 7:
            # temp_distance = np.sqrt(np.power(list_tracking[chanel][-1][0] - list_tracking[chanel][compare_index][0],2) + np.power(list_tracking[chanel][-1][1] - list_tracking[chanel][compare_index][1],2))
            if cX < seperate_line:
                if list_tracking[chanel][-15][1] > list_tracking[chanel][-10][1] and list_tracking[chanel][-10][1] > list_tracking[chanel][-5][1] and list_tracking[chanel][-5][1] > list_tracking[chanel][-1][1]:
                    reverse_flag = True
            else:
                if list_tracking[chanel][-15][1] < list_tracking[chanel][-10][1] and list_tracking[chanel][-10][1] < list_tracking[chanel][-5][1] and list_tracking[chanel][-5][1] < list_tracking[chanel][-1][1]:
                    reverse_flag = True

        # if list_tracking_flag[chanel] == False and len(list_tracking[chanel]) > number_of_step and abs(list_tracking[chanel][-1][1] - list_tracking[chanel][compare_index][1]) > distance_index:
        if list_tracking_flag[chanel] == False and len(list_tracking[chanel]) > number_of_step and reverse_flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.rectangle(frame, pt1, pt2, [0,0,255], 2)
            if is_enable_left and cX < seperate_line:
                if list_tracking[chanel][-1][1] < list_tracking[chanel][compare_index][1] :
                    cv2.imwrite(r'image_Log\reverse_detection_left_%d_%s.jpg' %(round(datetime.now().timestamp()*1000),threadID), frame)
                    bug_check = True
                    # thread = threading.Thread(target=send_image, args = (frame, '.jpg', oldest_file[8:21], 'WrongSide', [], '', 'None', name, lane_name)).start()
                    list_tracking_flag[chanel] = True
            elif is_enable_right and cX > seperate_line:
                if list_tracking[chanel][-1][1] > list_tracking[chanel][compare_index][1]:
                    cv2.imwrite(r'image_Log\reverse_detection_right_%d_%s.jpg' %(round(datetime.now().timestamp()*1000),threadID), frame)
                    bug_check = True
                    # thread = threading.Thread(target=send_image, args = (frame, '.jpg', oldest_file[8:21], 'WrongSide', [], '', 'None', name, lane_name)).start()
                    list_tracking_flag[chanel] = True


    #Create new track
    else: 
        flag = False 
        for i in range(len(list_tracking)):

            if list_tracking[i].size == 0 and cX != 0: #Handle empty list
                list_tracking[i] = np.append(list_tracking[i],[[cX,cY]],axis = 0)
                list_tracking_lenth[i] = list_tracking_lenth[i] + 1
                flag = True

                # if not sum(list_of_color[i]):
                #     list_of_color[i] = object_color
        if not flag:
            list_tracking.append(np.zeros(shape=(0,2)))
            list_tracking[-1] = np.append(list_tracking[-1],[[cX,cY]],axis = 0)#add to new track
            list_tracking_lenth.append(0)
            list_tracking_flag.append(False)
            list_tracking_lenth[-1] = list_tracking_lenth[-1] + 1

            # if not sum(list_of_color[len(list_tracking)-1]):
            #     list_of_color[len(list_tracking)-1] = object_color
    my_logger.debug(str(datetime.now()) + ' END clustering')                            
    return list_tracking, list_tracking_lenth, list_tracking_flag

def display_vehicle(frame, list_tracking, list_tracking_lenth, list_tracking_flag, count):
    for i in range(len(list_tracking)):
        # if is_add_list_tracking_lenth:
        #     for i in range(int(fps)):
        #         if (len(list_tracking[i]) > 0) or len(list_tracking[i]) > 20:
        #             list_tracking[i] = np.delete(list_tracking[i], 0, 0)
        #             if len(list_tracking[i]) == 0:
        #                 list_tracking_lenth[i] = 0
        #                 list_tracking_flag[i] = False
        # else:
        if (count >= 5 and len(list_tracking[i]) > 0) or len(list_tracking[i]) > 20:
            list_tracking[i] = np.delete(list_tracking[i], 0, 0)
            if len(list_tracking[i]) == 0:
                list_tracking_lenth[i] = 0
                list_tracking_flag[i] = False
        # for j in range(len(list_tracking[i])):
        #     if j > 0:
        #         frame = cv2.line(frame, (int(list_tracking[i][j-1][0]),int(list_tracking[i][j-1][1])),(int(list_tracking[i][j][0]),int(list_tracking[i][j][1])), [0,255,123], 3)
    return frame

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    # print(bbox)
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(name, lane_name, detections, img, list_tracking, list_tracking_lenth, list_tracking_flag, threadID, oldest_file, fps):
    my_logger.debug(str(datetime.now()) + ' START cvDrawBoxes')
    vehicle_count = 0
    
    for detection in detections:
        name_tag = str(detection[0])
        if name_tag == 'car' or name_tag == 'bus' or name_tag == 'truck' or name_tag == 'motorbike':# or name_tag == 'person':
            vehicle_count += 1
    if 'check' in oldest_file and vehicle_count == 0:
        list_tracking = [np.zeros(shape=(0,2))]
        list_tracking_lenth = [0]
        list_tracking_flag = [False]
        return img, list_tracking, list_tracking_lenth, list_tracking_flag
    
    list_tracking, list_tracking_lenth, list_tracking_flag = add_new_array(list_tracking, list_tracking_lenth, list_tracking_flag, vehicle_count)
    
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        name_tag = str(detection[0])
#         for name_key, color_val in color_dict.items():
        if name_tag == 'car' or name_tag == 'bus' or name_tag == 'truck' or name_tag == 'motorbike':# or name_tag == 'person':
            color = [24, 245, 217]
            xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            vehicle_size = abs(xmax - xmin)
#             if car_size > 50:
            
            list_tracking, list_tracking_lenth, list_tracking_flag = clustering(name, lane_name, list_tracking, list_tracking_lenth, list_tracking_flag, pt1, pt2, color, vehicle_size, img, threadID, oldest_file, fps)
                
            # cv2.rectangle(img, pt1, pt2, color, 1)
            # cv2.putText(img,
            #             detection[0] +
            #             " [" + str(round(detection[1] * 100, 2)) + "]",
            #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             color, 2)
        else:
            continue    
    my_logger.debug(str(datetime.now()) + ' END cvDrawBoxes')
    return img, list_tracking, list_tracking_lenth, list_tracking_flag

def create_folder(path):

    try:
        os.mkdir(path)
    except OSError:
        print ("Directory %s was created" % path)
    else:
        print ("Successfully created the directory %s " % path)

def motion_detection(frame, static_back):
    
    try:

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # gray = cv2.GaussianBlur(frame, (21, 21), 0) 

        diff_frame = cv2.absdiff(static_back, frame) 

        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 

        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

        cnts,_ = cv2.findContours(thresh_frame,  
                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        for contour in cnts: 
            # print(cv2.contourArea(contour))
            if cv2.contourArea(contour) > 800:
                return True, frame
        return False, frame
    
    except Exception as e:
        print ("Unexpected error in motion_detection:", e)
        return True, frame

def collect_frame_call_yolo(name, lane_name, rtsp, threadID):

    cap = cv2.VideoCapture(rtsp)

    global resp_time, array_queue

    array_queue[threadID] = Queue()

    fps = int(ch_config['video_config']['fps']) #cap.get(cv2.CAP_PROP_FPS)

    print(int(fps))

    run_yolo = sub_Thread(name, lane_name, threadID, fps).start()

    static_back = None
    # optimizition by using motion detection
    contour_count = 0
    width = int(cap.get(3))
    height = int(cap.get(4))
    y = int(float(ch_config[threadID]['top'])* height)
    x = int(float(ch_config[threadID]['seperate_line'])* width)

    count = 1
    ret_count = 1

    while True:
        try:

            if count > 1000:
                count = 1
                if '7' in threadID:
                    resp = str(ping('172.24.62.73',verbose = False))
                    # resp_time = 'response time = ' + resp[len(resp)-16:len(resp)-7]
                    resp_time = 'Ping to 172.24.62.73 : ' + re.search('Times.*ms', resp).group(0)
                    print(datetime.now(), resp_time)

            ret, frame = cap.read()
            crop_frame = copy.copy(frame)

            if not ret:
                ret_count += 1
                if ret_count >= 50:
                    print('time to work', datetime.now())
                    cap = cv2.VideoCapture(rtsp)
                print('not ret')
                continue

            if eval(ch_config[threadID]['is_enable_crop']):
                if eval(ch_config[threadID]['is_enable_right']):
                    crop_frame = frame[y:height, x:width]
                elif eval(ch_config[threadID]['is_enable_left']):
                    crop_frame = frame[y:height, 0:x]

            ret_count = 1

            count += 1
            if count % int(ch_config['decrease_fps']['decrease']) == 1:
                print('decrease_fps')
                continue

            # Motion detection START----------------------
            # Converting color image to gray_scale image 
            gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY) 
            # gray = cv2.GaussianBlur(gray, (21, 21), 0) 
            if static_back is None: 
                static_back = gray 
                continue

            # os.remove(oldest_file)

            is_has_object, _ = motion_detection(gray, static_back)

            is_have_to_check = 'no'
            if not is_has_object:
                contour_count += 1
                if contour_count > int(fps)* int(ch_config['countour_time']['countour_second']):
                    is_have_to_check = 'check'
                else:
                    continue

            static_back = gray

            if is_has_object:

                object_image = image_contener('%d_%d_%s' %(round(datetime.now().timestamp()*1000), contour_count, is_have_to_check), frame)
                array_queue[threadID].put(object_image)

                queue_size = array_queue[threadID].qsize()

                # print(threadID, queue_size)

                if queue_size > 300:
                    print('array_queue.queue.clear() threadID = ', threadID)
                    array_queue[threadID].queue.clear()
                    # run_yolo_again = sub_Thread(threadID).start()

            contour_count = 0
            
        except Exception as e:
            print ("Unexpected error in collect_frame_call_yolo:", e)

def get_list_path(path):
    list_of_files = os.listdir(path)
    full_path = ["{0}/{1}".format(path,x) for x in list_of_files]
    return full_path

def YOLO(name, lane_name, threadID, fps):

    global lock_func
    global bug_check

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    # initial vehicle array list
    list_tracking = [np.zeros(shape=(0,2))]
    list_tracking_lenth = [0]
    list_tracking_flag = [False]

    # count to remove element in tracking array
    shorten_array_count = 0

    list_path = []
    # resp_time = ''
    # start_time = time.time()

    while True:
        try:

            object_image = array_queue[threadID].get()

            oldest_file = object_image.name

            frame = object_image.image

            if shorten_array_count > 5:
                shorten_array_count = 0
            shorten_array_count = shorten_array_count + 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            # frame_queue.put(frame_resized)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            # darknet_image_queue.put(darknet_image)
            
            while lock_func:
                time.sleep(0.001)
            lock_func = True
            # lock.acquire()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
            # lock.release()
            lock_func = False

            image, list_tracking, list_tracking_lenth, list_tracking_flag = cvDrawBoxes(name, lane_name, detections, frame_resized, list_tracking, list_tracking_lenth, list_tracking_flag, threadID, oldest_file, fps)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = display_vehicle(image, list_tracking, list_tracking_lenth, list_tracking_flag, shorten_array_count)
            if bug_check:
            	bug_check = False
            	# image = cv2.putText(image, resp_time, (10, 100) , cv2.FONT_HERSHEY_SIMPLEX, 1, [24, 245, 217], -1, cv2.LINE_AA)
            	cv2.imwrite(r'image_Log\reverse_detection_%d_%s.jpg' %(round(datetime.now().timestamp()*1000),threadID), image)
            	

            # cv2.imshow(threadID, image)
            # if cv2.waitKey(1) == ord('q'):
            #     break

        except Exception as e:
            print ("Unexpected error in YOLO loop:", e)

    cv2.destroyAllWindows()


# if __name__ == '__main__':

#     YOLO()

config_file = "./cfg/yolov4.cfg" 
weights_file = "./yolov4.weights" 
data_file = "./cfg/coco.data" 

network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights_file,
        batch_size=1
    )

# width = darknet.network_width(network)
# height = darknet.network_height(network)
# darknet_image = darknet.make_image(width, height, 3)

# create log file
# logging.basicConfig(level = logging.INFO,filename = r'log\log_%s.log' %datetime.now().strftime("%Y-%m-%d"), format='%(asctime)s : Line No. : %(lineno)d - %(message)s')
LOG_FILENAME = r'log\violation_log.out'
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)
# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(
              LOG_FILENAME, maxBytes=1024000, backupCount=50)

my_logger.addHandler(handler)

# define config
poly_config = configparser.ConfigParser()
poly_config.read(r"cfg\poly.cfg")
ch_config = configparser.ConfigParser()
ch_config.read(r"cfg\ch.cfg")

# collect pre-send image into queue
image_pre_send_queue = Queue(maxsize = 100)

# flag of sending to plate recognize
is_ready_to_send = True

# initial threading
# thread0 = main_Thread("高師大和平", 'rtsp://admin:zhanyh3837303@172.24.62.73/cam/realmonitor?channel=0&subtype=0', 'ch0_wide')
# thread1 = main_Thread("高師大和平", 'rtsp://admin:zhanyh3837303@172.24.62.73/cam/realmonitor?channel=1&subtype=0', 'ch1_wide')
thread2 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zhanyh3837303@172.24.62.73/cam/realmonitor?channel=2&subtype=1', 'ch2wide')
# thread3 = main_Thread("高師大和平", 'rtsp://admin:zhanyh3837303@172.24.62.73/cam/realmonitor?channel=3&subtype=0', 'ch3_wide')
# thread4 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zhanyh3837303@172.24.62.73/cam/realmonitor?channel=4&subtype=1', 'ch4wide')
thread5 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zhanyh3837303@172.24.62.73/cam/realmonitor?channel=5&subtype=1', 'ch5wide')
thread6 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zhanyh3837303@172.24.62.73/cam/realmonitor?channel=6&subtype=1', 'ch6wide')
thread7 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zhanyh3837303@172.24.62.74:554/cam/realmonitor?channel=1&subtype=1', 'ch7wide')
# thread8 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zhanyh3837303@172.24.62.74:554/cam/realmonitor?channel=4&subtype=1', 'ch8wide')

# thread9 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zyh3837303@172.24.62.72:554/main_0', 'ch9wide')
thread10 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zyh3837303@172.24.62.72:554/main_1', 'ch10wide')
# thread11 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zyh3837303@172.24.62.72:554/main_2', 'ch11wide')
# thread12 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zyh3837303@172.24.62.72:554/main_3', 'ch12wide')
# thread13 = main_Thread("高師大燕巢", '汽車入口一', 'rtsp://admin:zhanyh3837303@172.24.62.78:554/1/h264major', 'ch13wide')




# thread0.start()
# thread1.start()
thread2.start()
# thread3.start()
# thread4.start()
thread5.start()
thread6.start()
thread7.start()
# thread8.start()

# thread9.start()
thread10.start()
# thread11.start()
# thread12.start()
# thread13.start()