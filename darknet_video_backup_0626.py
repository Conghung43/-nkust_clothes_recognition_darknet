from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from datetime import datetime
import numpy as np
import utils.class_define as cl
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime 
import logging.handlers
import inspect

array_queue = {}

def motion_detection(frame, static_back):
    try:
        h, w, _ = frame.shape
        frame = cv2.resize(frame,(int(h/2),int(w/2)),interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        gray = cv2.GaussianBlur(gray, (21, 21), 0) 
        if static_back is None:
            return True, gray
        diff_frame = cv2.absdiff(static_back, gray) 
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
        cnts,_ = cv2.findContours(thresh_frame,  
                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        for contour in cnts: 
            # print(cv2.contourArea(contour))
            if cv2.contourArea(contour) > 800:
                # print('had motion.')
                return True, gray
        return False, gray
    except Exception as e:
        print ("Unexpected error in motion_detection:", e)
        return True, gray

def get_subconfig_data(camera_name):
    for camera_configs in json_objets:
        type_detect = camera_configs.typeName
        for camera_config in camera_configs.cameraIdentifies:
            try:
                if camera_name == camera_config.camera.videoStreams[0].source:
                    coordinates = camera_config.identifyAreas[0].coordinate
                    start_time = camera_config.identifyTimes[0].startTime
                    end_time = camera_config.identifyTimes[0].endTime
                    type_id = camera_config.identifyTypes[0].id
                    camera_uuid = camera_config.camera.uuid
                    print('get_subconfig_data return')
                    return type_detect, coordinates, start_time, end_time, type_id, camera_uuid
            except Exception as ex:
                print('unexpected exception in get_subconfig_data', ex)

def get_subconfig_data_wear(camera_name):
    for camera_configs in json_objets:
        for camera_config in camera_configs.cameraIdentifies:
            try:
                if camera_name == camera_config.camera.videoStreams[0].source:
                    camera_uuid = camera_config.camera.uuid
                    return camera_uuid
            except Exception as ex:
                print('unexpected exception in get_subconfig_data', ex)

def display_vehicle(frame, object_data):
    for i in range(len(object_data.trajectory)):
        for j in range(len(object_data.trajectory[i])):
            if j > 0:
                frame = cv2.line(frame, (int(object_data.trajectory[i][j-1][0]),int(object_data.trajectory[i][j-1][1])),(int(object_data.trajectory[i][j][0]),int(object_data.trajectory[i][j][1])), [0,255,123], 3)
    return frame
function_lock = False
def yolo(camera_name, network, class_names, is_wear):
    global array_queue, function_lock, camera_running_list
    function_name = inspect.currentframe().f_code.co_name
    my_logger.info(str(datetime.now()) + ' START ' + function_name  + ' camera_name: ' + camera_name)
    print('yolo running........ ', camera_name)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    object_data = cl.object_character()
    fps = 30
    if is_wear:
        camera_uuid = get_subconfig_data_wear(camera_name)
    else:
        sub_config = get_subconfig_data(camera_name)
        _, _, start_time, end_time, _, _ = sub_config
        start_time = datetime.strptime(start_time, '%H:%M:%S').time()
        end_time = datetime.strptime(end_time, '%H:%M:%S').time()
    while True:
        try:
            if self_checking:
                if camera_name in camera_running_list:
                    if is_wear:
                        continue
                    sub_config = get_subconfig_data(camera_name)
                    _, _, start_time, end_time, _, _ = sub_config
                    start_time = datetime.strptime(start_time, '%H:%M:%S').time()
                    end_time = datetime.strptime(end_time, '%H:%M:%S').time()
                else:
                    print('End ', camera_name)
                    my_logger.info(str(datetime.now()) + ' self_checking ' + function_name  + ' camera_name: ' + camera_name + ' END' )
                    break
            object_image = array_queue[camera_name].get()
            current_time = datetime.now().time()
            frame_info = object_image.name.split('_')
            fps = int(frame_info[0])
            try:
                image_captured_time
            except:
                image_captured_time = frame_info[1]
            if int(frame_info[1]) - int(image_captured_time) > 5 and is_wear:
                image_captured_time = frame_info[1]
                continue
            frame = object_image.image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                        interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            while function_lock:
                time.sleep(0.01)
            function_lock = True
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
            function_lock = False
            if is_wear:
                image = darknet.draw_boxes_clothes(detections, frame_resized, camera_uuid)
            else:
                image, object_data = darknet.draw_boxes(detections, frame_resized, object_data, sub_config, frame_info, fps)
                image = display_vehicle(image, object_data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if is_wear:
                target = 'Clothes '
            else:
                target = 'Human '
            cv2.imshow(target + camera_name, image)
            if cv2.waitKey(1) == ord('q'):
                break
        except Exception as e:
            print("Unexpect exception in yolo: ", e)
            my_logger.error(str(datetime.now()) + ' Exception ' + function_name  + ' camera_name: ' + camera_name + str(e))
    my_logger.info(str(datetime.now()) + ' END ' + function_name  + ' camera_name: ' + camera_name)
    cv2.destroyAllWindows()

def get_json_object_from_file():
    import json
    from types import SimpleNamespace
    json_file =  open(r'C:\Users\Administrator\Documents\http_server\data.json', encoding="utf8") 
    data = json.load(json_file)
    # Change object to json string 
    data = json.dumps(data)
    # Parse JSON into an object with attributes corresponding to dict keys.
    data = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
    return data

def get_json_object(json_data):
    import json
    from types import SimpleNamespace
    # json_file =  open(r'C:\Users\Administrator\Documents\http_server\data.json', encoding="utf8") 
    # data = json.loads(str(json_data, encoding='utf-8'))
    # Change object to json string 
    # data = json.dumps(data)
    # Parse JSON into an object with attributes corresponding to dict keys.
    data = json.loads(json_data, object_hook=lambda d: SimpleNamespace(**d))
    return data

def read_camera_and_call_yolo(camera_config, network, class_names, is_wear):
    global array_queue, camera_running_list
    function_name = inspect.currentframe().f_code.co_name
    camera_name = camera_config.camera.videoStreams[0].source
    my_logger.info(str(datetime.now()) + ' START ' + function_name  + ' camera_name: ' + camera_name)
    array_queue[camera_name] = Queue() #initial queue
    print('read_camera_and_call_yolo running........ ', camera_name)
    cap = cv2.VideoCapture(camera_name)
    fps = int(cap.get(5))
    Thread(target=yolo, args=(camera_name, network, class_names, is_wear)).start()
    if not is_wear:
        _, _, start_time, end_time, _, _ = get_subconfig_data(camera_name)
    # start_time = datetime.strptime(start_time, '%H:%M:%S').time()
    # end_time = datetime.strptime(end_time, '%H:%M:%S').time()
    start_time = datetime.strptime('1:00:00', '%H:%M:%S').time()
    end_time = datetime.strptime('23:00:00', '%H:%M:%S').time()
    static_back = None
    ret_count = 1
    contour_count = 0
    while True:
        try:
            if self_checking:
                if camera_name in camera_running_list:
                    if is_wear:
                        continue
                    _, _, start_time, end_time, _, _ = get_subconfig_data(camera_name)
                    start_time = datetime.strptime(start_time, '%H:%M:%S').time()
                    end_time = datetime.strptime(end_time, '%H:%M:%S').time()
                else:
                    print('End ', camera_name)
                    break
            ret, frame = cap.read()
            if not ret:
                ret_count += 1
                if ret_count >= 50:# and datetime.now().hour < 21:
                    print('time to work ', camera_name, datetime.now())
                    my_logger.info(str(datetime.now()) + ' Reconnect camera ' + camera_name)
                    try:
                        cap.release()
                        cap = cv2.VideoCapture(camera_name)
                        ret_count = 1
                        my_logger.info(str(datetime.now()) + ' Reconnect success! ' + camera_name)
                    except:
                        print('Reconnect fail!')
                        my_logger.error(str(datetime.now()) + ' Reconnect fail ' + camera_name)
                        time.sleep(10)
                print('not ret', ret_count)
                continue
            current_time = datetime.now().time()
            ret_count = 1
            # Motion detection START----------------------
            have_motion, static_back = motion_detection(frame, static_back)
            check_ramdomly = False
            if not have_motion:
                contour_count += 1
                my_logger.debug(str(datetime.now()) + ' Check motion in camera ' + camera_name + 'contour_count = ' + str(contour_count) + 'fps*30 = ' + str(fps*30))
                if contour_count > (fps*30):# setup config fps*countour_second
                    my_logger.info(str(datetime.now()) + ' Check motion in camera ' + camera_name)
                    check_ramdomly = 'check'
                    contour_count = 0
                else:
                    continue
            # cv2.imshow(camera_name, frame)
            # if cv2.waitKey(1) == ord('q'):
            #     break
            object_image = cl.image_container('%d_%d_%d_%s' %(fps,
                    time.time(), 
                    contour_count, 
                    check_ramdomly), 
                    frame)
            array_queue[camera_name].put(object_image)
        except Exception as e:
            print ("Unexpected error in read_camera_and_call_yolo:", e)
            my_logger.error(str(datetime.now()) + ' Exception ' + function_name  + ' camera_name: ' + camera_name + str(e))
    print('Finished collect_frame_call_yolo')
    my_logger.info(str(datetime.now()) + ' END ' + function_name  + ' camera_name: ' + camera_name)

camera_running_list = []
self_checking = False
network, class_names, class_colors = darknet.load_network('./cfg/yolov4.cfg', 
'./cfg/coco.data','yolov4.weights' ,batch_size=1)
network_clothes, class_names_clothes, class_colors_clothes = darknet.load_network('./data/yolo-obj.cfg', './data/obj.data','yolo-clothes.weights' ,batch_size=1)
json_objets ={}
# Initial Log
LOG_FILENAME = r'log\logging.out'
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(
              LOG_FILENAME, maxBytes=10240000, backupCount=100)
my_logger.addHandler(handler)

def get_ms_since_start(start=False):
    global start_ms
    cur_time = datetime.now()
    # I made sure to stay within hour boundaries while making requests
    ms = cur_time.minute*60000 + cur_time.second*1000 + int(cur_time.microsecond/1000)
    if start:
        start_ms = ms
        return 0
    else:
        return ms - start_ms

class MyServer(BaseHTTPRequestHandler, object):
    def do_GET(self):
        print ("Start get method at %d ms" % get_ms_since_start(True))
        field_data = self.path
        self.send_response(200)
        self.end_headers()
        with open('receive_data_get.npy', 'wb') as f:
            np.save(f, field_data, allow_pickle=True)
            f.close()
        # self.wfile.write(str(field_data))
        print ("Sent response at %d ms" % get_ms_since_start())
        return

    def do_POST(self):
        global camera_running_list, self_checking, json_objets
        function_name = inspect.currentframe().f_code.co_name
        my_logger.info(str(datetime.now()) + ' START ' + function_name)
        print ("Start post method at %d ms" % get_ms_since_start(True))
        length = int(self.headers.get('Content-Length'))
        print ("Length to read is %d at %d ms" % (length, get_ms_since_start()))
        field_data = self.rfile.read(length)
        print ("Reading rfile completed at %d ms" % get_ms_since_start())
        self.send_response(200)
        self.end_headers()
        
        temp_camera_running_list = []
        json_objets = get_json_object(field_data)
        f = open('log/{}.txt'.format(time.time()), "a")
        for json_objet in json_objets:
            for camera_config in json_objet.cameraIdentifies:
            # camera_config = json_objet.cameraIdentifies[0]
                camera_name = camera_config.camera.videoStreams[0].source
                f.write(camera_name + '\n')
                return_data = get_subconfig_data(camera_name)
                if json_objet.typeName == 'wear':
                    # Thread(target=read_camera_and_call_yolo, args=(camera_config, network_clothes, class_names_clothes, True)).start()
                    Thread(target=read_camera_and_call_yolo, args=(camera_config, network, class_names, True)).start()
                    temp_camera_running_list.append(camera_name)
                if not return_data:
                    continue
                if camera_name not in camera_running_list:
                    Thread(target=read_camera_and_call_yolo, args=(camera_config, network, class_names, False)).start()
                temp_camera_running_list.append(camera_name)
                break
        camera_running_list = temp_camera_running_list
        f.close()
        self_checking = True
        time.sleep(10)
        self_checking = False
        print ("Sent response at %d ms" % get_ms_since_start())
        my_logger.info(str(datetime.now()) + ' END ' + function_name)
        return

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), MyServer)
    print ('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()