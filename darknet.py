#!python3
"""
Python 3 wrapper for identifying objects in images
Requires DLL compilation
Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".
On a GPU system, you can force CPU evaluation by any of:
- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"
Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)
Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py
@author: Philip Kahn
@date: 20180503
"""
from ctypes import *
import math
import random
import os
import threading
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import json
import requests
from datetime import datetime
import ast
import utils.class_define as cl
import base64
from io import BufferedReader, BytesIO
import cv2
from sklearn.cluster import KMeans
import time
from ast import literal_eval
from utils import get_color_two_side_clothes as color_handle
from threading import Thread
import inspect

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 
        0, 
        batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))

def send_image(image, occurred_time, key, model, types, cameraUUID, color_vector_list):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('image_log/image_{}.jpg'.format(str(time.time())), image)
        # return
        ret, img_encode = cv2.imencode('.jpg', image)
        str_encode = img_encode.tobytes()
        byte_io = BytesIO(str_encode)
        byte_io.name = 'picture.jpg'
        buffer_reader = BufferedReader(byte_io)

        myobj = {
                    "key" :  key,
                    "model" : model,
                    "time": occurred_time,
                    "types": types,
                    "cameraUUID": cameraUUID, 
                    "featureVecture": color_vector_list}
        response = requests.post('http://163.18.18.69:3009/v1/image-recognition-api', json = myobj)

        UUID = str(response.content, 'utf-8')[1:-1]
        print(UUID)
        myobj = {
                    "uuid" : UUID,
                    "key": key}
        response = requests.post('http://163.18.18.69:3009/v1/image-recognition-api/image', files = {'files':buffer_reader}, data= myobj)
        print(' END send_image ' + str(response.status_code))
    except Exception as e:
        print(" Unexpected error in send_image: "+ str(e))

def polygon(label,image):

    for conf in poly_config[label]:
        temp_array.append(poly_config[label][conf])

    for i in range(0,len(temp_array),2):
        coordinate_array.append((int(float(temp_array[i])*image.shape[1]),int(float(temp_array[i+1])*image.shape[0])))

    polygon = Polygon(coordinate_array)

    point = Point(i, j)
    if not polygon.contains(point):
        return True
    else:
        return False

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

def add_new_array(object_data, total_object):
    for i in range(total_object):
        if len(object_data.trajectory) < total_object:
            object_data.trajectory.append(np.zeros(shape=(0,2)))
            object_data.trajectory_length.append(0)
            object_data.flag.append(False)
            object_data.is_capture.append(False)
    return object_data

def sizes2points(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def send_image2another_model(image):
    try:
        ret, img_encode = cv2.imencode('.jpg', image)
        str_encode = img_encode.tobytes()
        response = requests.post('http://127.0.0.1:8002/', data=str_encode, timeout=12)
        image_info = response.reason
        image_info = literal_eval(image_info)
        print(image_info)
        return image_info
    except Exception as e:
        print(" Unexpected error in send_image: " + str(e))
        return ''

def get_crop_extend(pt1, pt2, ori_h, ori_w):

    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    w,h = pt2 - pt1
    pt1_temp = 2*pt1 - pt2
    pt2_temp = 2*pt2 - pt1
    pt1 = pt1_temp
    pt2 = pt2_temp
    if pt1[0] < 0:
        pt1[0] = 0
    if pt1[1] < 0:
        pt1[1] = 0
    if pt2[0] > ori_w:
        pt2[0] = ori_w
    if pt2[1]  > ori_h:
        pt2[1] = ori_h

    return pt1, pt2

# def send_image_to_mask_server(image, camera_uuid):
#     try:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         ret, img_encode = cv2.imencode('.jpg', image)
#         str_encode = img_encode.tobytes()
#         send_data = str.encode(camera_uuid + '=') + str_encode
#         response = requests.post('http://203.64.105.136:8005/', data=send_data, timeout=12)
#         image_info = response.reason.split(';')
#         image_info = [info.split('_') for info in image_info]
#         image_info = [(info[0], info[1], [float(value) for value in info[2][1:-1].split(',')]) for info in image_info if len(info) == 3]
#         return image_info
#     except Exception as e:
#         print(" Unexpected error in send_image: " + str(e))
#         return ''
def send_image_to_mask_server(image, camera_uuid):
    try:
        # header = {'cam_id':camera_uuid}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret, img_encode = cv2.imencode('.jpg', image)
        str_encode = img_encode.tobytes()
        # send_data = str.encode(camera_uuid + '=') + str_encode
        # send_data = json.dumps(send_data)
        response = requests.post('http://203.64.105.136:8010/', data=str_encode, timeout=12)
        image_info = response.reason.split(' ')
        image_info = np.reshape(np.array(image_info).astype(np.float),(6,16)).tolist()

        return image_info
    except Exception as e:
        print(" Unexpected error in send_image: " + str(e))
        return []

def clustering(object_data, pt1, pt2, object_color, bounding_box_size, frame, person_count, sub_config, fps, my_logger):
    function_name = inspect.currentframe().f_code.co_name
    try:
        if type(sub_config) is str:
            mode_name = 'wear'
            area = [[0.1,0.4],[0.9,0.4],[0.9,0.9],[0.1,0.9]]
            area = np.array([x*np.array(frame.shape)[:2] for x in area]).astype(int)
            camera_uuid = sub_config
        else:
            mode_name, area, _, _, mode_no, camera_uuid = sub_config

            area = area.replace('{"x":','[').replace('"y":','').replace('}',']')
            area = ast.literal_eval(area)

            area = np.array([x*np.array(frame.shape)[:2] for x in area]).astype(int)
        my_logger.debug(str(datetime.now()) + ' START ' + function_name  + ' camera_name: ' + camera_uuid)
        # for point in area:
        #     frame = cv2.circle(frame,tuple(point), 5, (0,0,255), -1)

        cX, cY = (np.array(pt1) + np.array(pt2))/2
        # cY = pt2[1]
        chanel = find_nearest_point( cX, cY, object_data.trajectory, bounding_box_size)
        polygon = Polygon(area)

        if chanel >= 0:

            # print(chanel)

            # if cY < 0.1*frame.shape[0] or cY > 0.9*frame.shape[0] or cX < 0.1*frame.shape[1] or cX > 0.9*frame.shape[1]:
            #     object_data.trajectory[chanel] = np.zeros(shape=(0,2))
            #     object_data.trajectory_length[chanel] = 0
            #     object_data.flag[chanel] = False
            #     object_data.is_capture[chanel] = False
            #     return object_data

            if pt1[1] < 0.1*frame.shape[0] or pt2[1] > 0.9*frame.shape[0] or pt1[0] < 0.1*frame.shape[1] or pt2[0] > 0.9*frame.shape[1]:
                object_data.trajectory[chanel] = np.zeros(shape=(0,2))
                object_data.trajectory_length[chanel] = 0
                object_data.flag[chanel] = False
                object_data.is_capture[chanel] = False
                return object_data

            object_data.trajectory_length[chanel] += 1
            
            point = Point([cX, cY])

            in_poly = polygon.contains(point)

            if in_poly:
                object_data.flag[chanel] = True

            if not object_data.is_capture[chanel] and object_data.flag[chanel]:
                occurred_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h, w, _ = frame.shape
                if mode_name is 'wear':
                    my_logger.debug(str(datetime.now()) + ' clustering, mode_name is wear ' + occurred_time + mode_name + camera_uuid)
                    object_data.is_capture[chanel] = True
                    people_crop =  frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    cv2.imwrite('image_log/crop_image_{}.jpg'.format(str(time.time())), people_crop)
                    color_vector_list = send_image_to_mask_server(people_crop, camera_uuid)
                    color_list = color_handle.get_surrounded_color(people_crop, pt1,pt2)
                    # pt1, pt2 = get_crop_extend(pt1,pt2,h,w)
                    frame_crop = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    type_data = send_image2another_model(frame_crop)

                    top_list = ["short sleeve top","long sleeve top","short sleeve outwear","long sleeve outwear","vest","sling"]
                    button_list = ["shorts","trousers","skirt","short sleeve dress","long sleeve dress","vest dress","sling dress"]
                    # type_data = []
                    if len(type_data) == 0:
                        type_data = [{'type':'short sleeve top', 'proportion': 100, 'colors': [color_handle.rgb_to_hex(color_list[0])]}]
                        if len(color_list) > 1:
                            type_data.append({'type':'trousers', 'proportion': 100, 'colors': [color_handle.rgb_to_hex(color_list[1])]})
                    else:
                        top_exist = False
                        for type_dt in type_data:
                            if type_dt['type'] in top_list:
                                color_diff = color_handle.color_diff(color_handle.hex_to_rgb(type_dt['colors'][0]), color_list[0],20) 
                                if color_diff:
                                    type_dt['colors'].append(color_handle.rgb_to_hex(color_list[0]))
                                top_exist = True
                        if not top_exist:
                            type_data.append({'type':'short sleeve top', 'proportion': 100, 'colors': [color_handle.rgb_to_hex(color_list[0])]})
                        # send_image(frame, occurred_time, "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", mode_name, type_data, camera_uuid, color_vector_list)
                        Thread(target=send_image, args= (frame, 
                                                        occurred_time, 
                                                        "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", 
                                                        mode_name, 
                                                        type_data, 
                                                        camera_uuid, 
                                                        color_vector_list)).start()
                    my_logger.debug(str(datetime.now()) + 'wear sent!' + str(type_data))
                    cv2.imwrite('image_log/wear_image_{}.jpg'.format(str(time.time())), frame)
                elif mode_name in ["wander", "stay"]:
                    if object_data.trajectory_length[chanel] > 300:# set up time in config
                        my_logger.debug(str(datetime.now()) + ' clustering, mode_name is not wear ' + occurred_time + mode_name + camera_uuid)
                        if in_poly:
                            print('Dou liu')
                            # send_image(frame, occurred_time, "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", mode_name, [{'type':'people', 'proportion':100, 'colors': []}] , camera_uuid, [])
                            Thread(target=send_image, args= (frame, 
                                                            occurred_time, 
                                                            "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", 
                                                            mode_name, 
                                                            [{'type':'people', 'proportion':100, 'colors': []}], 
                                                            camera_uuid, 
                                                            [])).start()
                        else:
                            print('paihui')
                            # send_image(frame, occurred_time, "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", mode_name, [{'type':'people', 'proportion':100, 'colors': []}], camera_uuid, [])
                            Thread(target=send_image, args= (frame, 
                                                            occurred_time, 
                                                            "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", 
                                                            mode_name, 
                                                            [{'type':'people', 'proportion':100, 'colors': []}], 
                                                            camera_uuid, 
                                                            [])).start()
                            # Pai hui
                        object_data.is_capture[chanel] = True
                else:
                    if in_poly:
                        # take picture here
                        print('night invasion')
                        # send_image(frame, occurred_time, "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", mode_name, [{'type':'people', 'proportion':100, 'colors': []}], camera_uuid,[])
                        Thread(target=send_image, args= (frame, 
                                                        occurred_time, 
                                                        "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", 
                                                        mode_name, 
                                                        [{'type':'people', 'proportion':100, 'colors': []}], 
                                                        camera_uuid, 
                                                        [])).start()
                        object_data.is_capture[chanel] = True

            object_data.trajectory[chanel] = np.append(object_data.trajectory[chanel],[[cX,cY]],axis = 0)

        else: 
            flag = False 
            for i in range(len(object_data.trajectory)):

                if object_data.trajectory[i].size == 0 and cX != 0: #Handle empty list
                    object_data.trajectory[i] = np.append(object_data.trajectory[i],[[cX,cY]],axis = 0)
                    object_data.trajectory_length[i] = 0
                    object_data.flag[i] = False
                    flag = True

            if not flag:
                object_data.trajectory.append(np.zeros(shape=(0,2)))
                object_data.trajectory[-1] = np.append(object_data.trajectory[-1],[[cX,cY]],axis = 0)#add to new track
                object_data.trajectory_length.append(0)
                object_data.trajectory_length[-1] = 0
                object_data.flag.append(False)
                object_data.flag[-1] = False
                object_data.is_capture.append(False)
                object_data.is_capture[-1] = False
    except Exception as ex:
        my_logger.debug(str(datetime.now()) + ' ERROR ' + function_name  + ' camera_name: ' + camera_uuid + str(ex))
    my_logger.debug(str(datetime.now()) + ' END ' + function_name  + ' camera_name: ' + camera_uuid + ' Thread count = ' + str(threading.activeCount()))
    return object_data

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        #read image
        img = self.IMAGE
        try:
            #convert to rgb from bgr
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except: 
            pass
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)

def find_couple(current_box, target_box):
    x, y, w, h = current_box
    c_xmin, c_ymin, c_xmax, c_ymax = sizes2points(float(x), float(y), float(w), float(h))
    x, y, w, h = target_box
    t_xmin, t_ymin, t_xmax, t_ymax = sizes2points(float(x), float(y), float(w), float(h))

    if x > c_xmin and x < c_xmax and (t_ymin < c_ymax or t_ymax > c_ymin):
        if c_ymin < t_ymin:
            ymin = c_ymin
            ymax = t_ymax
        else:
            ymin = t_ymin
            ymax = c_ymax
        if c_xmin < t_xmin:
            xmin = c_xmin
        else:
            xmin = t_xmin
        if c_xmax > t_xmax:
            xmax = c_xmax
        else:
            xmax = t_xmax

        return True, [xmin,ymin,xmax,ymax]
    return [False]

def get_object_color(detections, image):
    object_colors = []
    frame = image.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for detection in detections:
        x, y, w, h = detection[2]
        # xmin, ymin, xmax, ymax = sizes2points(float(x), float(y), float(w), float(h))
        # crop_img = frame[ymin:ymax, xmin:xmax]
        # cv2.imshow('image', crop_img)
        # cv2.waitKey(5000)
        # dc = DominantColors(crop_img, 5) 
        # colors = dc.dominantColors()
        color = image[int(y)][int(x)]
        hex_list = '#%02x%02x%02x' % tuple([color[2],color[1],color[0]])
        object_colors.append(hex_list)
    return object_colors
# Test get_object_color
# detections = [['ao','90',[29,93,10,10]]]
# image = cv2.imread(r"C:\Users\Administrator\Documents\April_project\people\image_log\image_1623161178.3000145.jpg")
# get_object_color(detections, image)
def draw_boxes_clothes(detections, frame, camera_uuid):

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    object_color = get_object_color(detections, frame)
    detect_clothes = []
    for index, detection in enumerate(detections):
        label, acc, current_box = detection
        if current_box[3]/current_box[2] > 3:
            continue
        for i in range(index + 1, len(detections), 1):
            t_label, t_acc, target_box = detections[i]
            if target_box[3]/target_box[2] > 3:
                continue
            check2box_status = find_couple(current_box, target_box)
            if check2box_status[0]:
                xmin,ymin,xmax,ymax = check2box_status[1]
                # crop_img = frame[ymin:ymax, xmin:xmax]
                # cv2.imshow('image', crop_img)
                # cv2.waitKey(5000)
                hex_list = [object_color[index], object_color[i]]
                print(' hex_list....................', hex_list)
                occurred_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_image(frame, 
                            occurred_time, 
                            "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", 
                            'wear', 
                            [{'type':label, 'colors': [object_color[index]]}, {'type':t_label, 'colors': [object_color[i]]}], 
                            camera_uuid)
                break
            if i == len(detections) - 1:
                x, y, w, h = current_box
                xmin, ymin, xmax, ymax = sizes2points(float(x), float(y), float(w), float(h))
                # crop_img = frame[ymin:ymax, xmin:xmax]
                # cv2.imshow('image', crop_img)
                # cv2.waitKey(5000)
                hex_list = [object_color[index]]
                print(' hex_list....................', hex_list)
                occurred_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_image(frame, 
                            occurred_time, 
                            "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", 
                            'wear', 
                            [{'type':label, 'colors': [object_color[index]]}], 
                            camera_uuid)
                            
    return frame

# frame = cv2.imread(r"C:\Users\Administrator\Documents\darknet_clothes\build\darknet\x64\data\obj\024338.jpg")
# detections = [['shorts','100',[470,706,200,260]],['shorts','98',[480,999,150,380]]]
# draw_boxes_clothes(detections, frame, '2677d98d-873d-4168-b05f-29a809935830')

def draw_boxes(detections, frame, object_data, sub_config, oldest_file, fps, my_logger):
    function_name = inspect.currentframe().f_code.co_name
    if type(sub_config) is str:
        camera_uuid = sub_config
    else:
        _, _, _, _, _, camera_uuid = sub_config
    my_logger.debug(str(datetime.now()) + ' START ' + function_name  + ' camera_name: ' + camera_uuid)
    try:
        # print('draw_boxes')
        person_count = 0
        for detection in detections:
            name_tag = str(detection[0])
            if name_tag == 'person':
                person_count += 1
        if 'check' in oldest_file and person_count == 0:
            # imwritr image here with sub_config[camera_id]
            # if type(sub_config) is str:
            #     camera_uuid = sub_config
            # else:
            #     _, _, _, _, _, camera_uuid = sub_config
            # color_handle.save_image_base_cam_ui(camera_uuid, frame)
            object_data = cl.object_character()
            return frame, object_data
        object_data = add_new_array(object_data, person_count)
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            name_tag = str(detection[0])
            if w < 50:
                continue
            if name_tag == 'person':
                color = [24, 245, 217]
                xmin, ymin, xmax, ymax = sizes2points(
                float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                people_width = int(abs(xmax - xmin))
                people_height = int(abs(ymax - ymin))
                y_center = int((pt1[1] + pt2[1])/2)
                x_center = int((pt1[0] + pt2[0])/2)
                frame_copy = frame.copy()
                frame_copy = cv2.rectangle(frame_copy, pt1, pt2, color, 1)
                object_data = clustering(object_data, pt1, pt2, [24, 245, 217], 50, frame_copy, person_count, sub_config, fps, my_logger)

                # frame = cv2.rectangle(frame, pt1, pt2, color, 1)
                # frame = cv2.putText(frame, "w= {} h= {}".format(int(people_width), int(people_height)),
                #             (xmin, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             color, 2)
    except Exception as e:
        print(" Unexpected error in draw_boxes: "+ str(e))
        my_logger.debug(str(datetime.now()) + ' ERROR ' + function_name  + ' camera_name: ' + camera_uuid + str(e))
    my_logger.debug(str(datetime.now()) + ' END ' + function_name  + ' camera_name: ' + camera_uuid)
    return frame, object_data


def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])


#  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)