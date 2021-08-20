import json
import requests
import cv2
import base64
from io import BufferedReader, BytesIO
import uuid
import time
import inspect
from datetime import datetime

from ctypes import *
import math
import os
import numpy as np
from collections import Counter


# def send_image(image):
#     try:
#         ret, img_encode = cv2.imencode('.jpg', image)
#         str_encode = img_encode.tostring()
#         response = requests.post('http://127.0.0.1:8000/', data= str_encode)
#         print(' END send_image ' + str(response.status_code))
#     except Exception as e:
#         print(" Unexpected error in send_image: "+ str(e))

def send_image(image):
    try:
        ret, img_encode = cv2.imencode('.jpg', image)
        str_encode = img_encode.tobytes()
        response = requests.post('http://127.0.0.1:8000/', data=str_encode, timeout=12)
        image_info = response.reason.split(';')
        image_info = [info.split('_') for info in image_info]
        image_info = [(info[0], info[1], [float(value) for value in info[2][1:-1].split(',')]) for info in image_info if len(info) == 3]
        return image_info
    except Exception as e:
        print(" Unexpected error in send_image: " + str(e))
        return ''

def send_image(image, occurred_time, key, model, types, cameraUUID, color_vector):
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
                    "featureVecture": color_vector}
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


import json
import numpy as np
from ast import literal_eval
def send_image2another_model(image):
    try:
        ret, img_encode = cv2.imencode('.jpg', image)
        str_encode = img_encode.tobytes()
        response = requests.post('http://127.0.0.1:8002/', data=str_encode, timeout=12)
        image_info = response.reason
        image_info = literal_eval(image_info)
        return image_info
    except Exception as e:
        print(" Unexpected error in send_image: " + str(e))
        return ''

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

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

# print(get_crop_extend([2,3], [30,50], 90, 80))

frame = cv2.imread(r"C:\Users\Administrator\Documents\April_project\people\dog - Copy.jpg")
color_vector = send_image_to_mask_server(frame,'1234')
# print(unique_count_app(frame))
# cv2.imshow('image', frame)
# cv2.waitKey(0)
# print(str(send_image2another_model(image)))
# type_data = send_image2another_model(frame)
type_data = [{'type': 'trousers', 'proportion':100, 'colors': ['#a62e34']}]
# image_info = literal_eval(type_data)
# print(image_info)
occurred_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_image(frame, occurred_time, "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", 'wear', type_data, 'b1e5c3a8-8c76-4d0c-ad0d-b868d430a22e', color_vector)
# send_image(image, "2021-04-12 01:02:01", "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", "wear", [{'type':'vest', 'colors': ['#395697']}, {'type':'trousers', 'colors': ['#ff0202']}], '38f9ef50-6c54-4bcf-bae8-141687bab55b')        
# send_image(image, "2021-04-12 01:02:01", "82b13762-e993-43f1-b22c-8fdb4e5c5d8f", "wander", [{'type':'people', 'colors': []}], '38f9ef50-6c54-4bcf-bae8-141687bab55b')        
# send_image(frame, occurred_time, "1bca0da3-3560-4969-b5f3-cdc11e8c0647", mode_name, ["people"], camera_uuid
# import cv2
# cap = cv2.VideoCapture('rtsp://syno:57f97d86b9760ed4390619f36fc06ced@203.64.95.248:554/Sms=52.unicast')
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('image', frame)
#     cv2.waitKey(1)s

# image = cv2.imread('dog.jpg')
# print(send_image(image))