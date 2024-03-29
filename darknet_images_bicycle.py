#! /usr/bin/python3

import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet_original as darknet
from datetime import datetime
import glob
from http.server import BaseHTTPRequestHandler, HTTPServer

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    # image_infor = darknet.draw_boxes(detections, image_resized, class_colors)
    image_info = ''
    object_list = ['bicycle', 'motorbike', 'person', 'bus', 'truck','car']
    for label, acc, bbox in detections:
        if label in object_list:
            bbox = tuple(np.array(bbox)/width)
            image_info = image_info + label + '_' + acc + '_' + str(bbox) + ';'

    return image_info

def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    # print(detections)


args = parser()
check_arguments_errors(args)

random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
args.config_file,
args.data_file,
args.weights,
batch_size=args.batch_size
)

images = load_images(args.input)
# cap = cv2.VideoCapture('CH01Test.mp4')

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

        frame = cv2.imdecode(np.frombuffer(field_data, np.uint8), -1)
        image_info = image_detection(
            frame, network, class_names, class_colors, args.thresh
            )

        print("Reading rfile completed at %d ms" % get_ms_since_start())
        self.send_response(200, message=image_info)
        self.end_headers()
        # self.wfile.write(field_data)
        print("Sent response at %d ms" % get_ms_since_start())
        return


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8001), MyServer)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()


# for image_path in glob.glob("*.jpg"):
# 	frame = cv2.imread(image_path)
# 	start_time = datetime.now()
# 	image = image_detection(
# 	    frame, network, class_names, class_colors, args.thresh
# 	    )
# 	# print(detections)
# 	end_time = datetime.now()
# 	print(end_time - start_time)

# 	cv2.imshow('Inference', image)
# 	cv2.waitKey(0)
# cv2.destroyAllWindows()

