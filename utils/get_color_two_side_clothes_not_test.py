from numpy.lib.type_check import imag
from sklearn.cluster import KMeans
from math import *
import cv2

def get_colors(colors):
    est = KMeans(n_clusters=1)
    est.fit(colors)
    return est.cluster_centers_[0].astype(int)

def get_circle_pixel(center, r):
    return [[int(center[0] + cos(radians(index))*r), int(center[1] + sin(radians(index))*r)] for index in range(0,360,10)]

# circle = get_circle_pixel([100,100],10)

def color_diff(color1, color2, threadhold):
    r1,g1,b1 = color1
    r2,g2,b2 = color2
    distance = sqrt(pow(r1-r2,2)+pow(g1-g2,2)+pow(b1-b2,2))
    if distance > threadhold:
        return True
    else:
        return False

def get_main_color(image, template_pos, r):
    x, y = template_pos
    circle_pixel_list = get_circle_pixel([x,y], r)
    circle_color_list = [image[pixel_color[1]][pixel_color[0]] for pixel_color in circle_pixel_list]
    main_color = get_colors(circle_color_list)
    return main_color

def get_color_compare(image, template_pos, r, pt1, cam_id):
    color_1 = get_main_color(image, template_pos, r)
    background_image = red_image_base_cam_id(cam_id)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    template_pos = np.array(template_pos) + np.array(pt1)
    color_2 = get_main_color(background_image, template_pos, r)
    if color_diff(color_1, color_2,10):
        return color_1
    else:
        return ''

def get_surrounded_color(image, pt1, cam_id):
    h,w,_ = image.shape
    main_color_list = []
    if h/w < 2:
        main_color = get_main_color(image, [w/2, h/2], w/8, pt1, cam_id)
        if main_color != '':
            main_color_list.append(main_color)
    else:
        main_color = get_main_color(image, [w/2,h/4], w/8, pt1, cam_id)
        if main_color != '':
            main_color_list.append(main_color)

        main_color = get_main_color(image, [w/2,h*3/4], w/8, pt1, cam_id)
        if main_color != '':
            main_color_list.append(main_color)
    return main_color_list

def rgb_revert(rgb):
    return [rgb[2],rgb[1],rgb[0]]

def hex_to_rgb(hex_code):
    hex_code = hex_code[1:]
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return rgb
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)
# print(hex_to_rgb('#867565'))
# print(rgb_to_hex([134, 117, 101]))

def save_image_base_cam_ui(cam_ui, image):
    cv2.imwrite('image_log/{}.jpg'.format(cam_ui), image)

def red_image_base_cam_id(cam_ui):
    cv2.imread('image_log/{}.jpg'.format(cam_ui))