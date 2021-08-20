# Import packages
import ast
from http.server import BaseHTTPRequestHandler, HTTPServer
from utils import visualization_utils_plate_recognize as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import socket
import struct
import logging.handlers
from datetime import datetime

# os.environ['PYTHONPATH'] = ['C:/tensorflow1/models;C:/tensorflow1/models/research;C:/tensorflow1/models/research/slim']
# sys.path.append('C:/tensorflow1/models')
# sys.path.append('C:/tensorflow1/models/research')
# sys.path.append('C:/tensorflow1/models/research/slim')
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# Import utilites
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = '1.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 36

LOG_FILENAME = r'log\logging.out'

# Set up a specific logger with our desired output level
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(
    LOG_FILENAME, maxBytes=102400000, backupCount=100)

my_logger.addHandler(handler)


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)

    # sess = tf.compat.v1.Session(graph=detection_graph)
    sess = tf.compat.v1.Session(
        graph=detection_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

default_image = cv2.imread('img_5_150.png')

sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
         feed_dict={image_tensor: np.expand_dims(default_image, axis=0)})

data = b""
image_path = r"image_check_process_status\image_for_checking.jpg"


def get_ms_since_start(start=False):
    global start_ms
    cur_time = datetime.now()
    # I made sure to stay within hour boundaries while making requests
    ms = cur_time.minute*60000 + cur_time.second * \
        1000 + int(cur_time.microsecond/1000)
    if start:
        start_ms = ms
        return 0
    else:
        return ms - start_ms


def run_tf_recognize(receive_image):
    try:
        print('start run run_tf_recognize')
        frame = cv2.imdecode(np.frombuffer(receive_image, np.uint8), -1)
        # print(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        plate_character = vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        print('plate_character', plate_character)
        # print('end tf recognize, result = ', plate_character)
        return plate_character

    except Exception as e:
        my_logger.debug(str(datetime.now()) +
                        " Unexpected error in main:" + str(e))

class MyServer(BaseHTTPRequestHandler, object):
    def do_GET(self):
        print("Start get method at %d ms" % get_ms_since_start(True))
        field_data = self.path
        self.send_response(200)
        self.end_headers()
        # with open('receive_data_get.npy', 'wb') as f:
        #     np.save(f, field_data, allow_pickle=True)
        #     f.close()
        # self.wfile.write(str(field_data))
        print("Sent response at %d ms" % get_ms_since_start())
        return

    def do_POST(self):
        print("Start post method at %d ms" % get_ms_since_start(True))
        length = int(self.headers.get('Content-Length'))
        print("Length to read is %d at %d ms" % (length, get_ms_since_start()))
        field_data = self.rfile.read(length)

        # field_data = np.fromstring(field_data, dtype='uint8')
        # image = cv2.imdecode(field_data, flags=1)
        print('start tf recognize')
        plate_character = run_tf_recognize(field_data)
        print('end tf recognize', plate_character)

        print("Reading rfile completed at %d ms" % get_ms_since_start())
        self.send_response(200, message=plate_character)
        self.end_headers()
        # self.wfile.write(field_data)
        print("Sent response at %d ms" % get_ms_since_start())
        return


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), MyServer)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()
