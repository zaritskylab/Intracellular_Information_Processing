from skimage import io
import numpy as np
from tifffile import imsave
import cv2
import matplotlib.pyplot as plt
from pillars_utils import *
import random
from consts import *
import time

import numpy
from matplotlib.pyplot import axline
from pathlib import Path
from matplotlib import pyplot as plt
from Pillars.visualization import *
from Pillars.granger_causality import *
from Pillars.repositioning import *
from Pillars.analyzer import *
from pathlib import Path
from Pillars.runner_helper import *
from Pillars.runner import *

import json
import math


class Labeler:
    def __init__(self):
        self.centers = []
        self.lns = []
        self.alive_centers = []
        self.force_click = False

    def label_video(self, config_name):
        f = open("../configs/" + config_name)
        config_data = json.load(f)
        update_const_by_config(config_data, config_name)
        # self.alive_centers = []
        self.alive_centers = get_seen_centers_for_mask()

        self.img = get_last_image()
        # self.img = get_images(get_images_path())[0]
        # self.img = get_images(get_images_path())[60]
        self.print_alive_centers()

    def print_alive_centers(self):
        def onclick(event):
            x = int(event.ydata)
            y = int(event.xdata)

            if not self.force_click:
                repositioned_clicked_location = get_center_fixed_by_circle_mask_reposition((x,y),
                                                                                           self.img, 5)
            else:
                repositioned_clicked_location = (x,y)

            self.lns.append(plt.plot([repositioned_clicked_location[1]], [repositioned_clicked_location[0]], ls='none', marker='.', ms=5, c='red'))
            self.centers.append(repositioned_clicked_location)
            plt.show()

        def close_handler(event):
            self.fix_centers()

        def key_press_event(event):
            if event.key == 'z':
                if len(self.centers) > 0:
                    self.centers.pop()
                    last_added_point = self.lns.pop()
                    last_added_point = last_added_point.pop()
                    if last_added_point is not None:
                        last_added_point.remove()
                        plt.show()
            elif event.key == 'x':
                self.force_click = not self.force_click

        fig, ax = plt.subplots()
        plt.imshow(self.img, cmap=plt.cm.gray)
        y = [center[0] for center in self.alive_centers]
        x = [center[1] for center in self.alive_centers]
        scatter_size = [3 for center in self.alive_centers]
        plt.scatter(x, y, s=scatter_size)
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('close_event', close_handler)
        fig.canvas.mpl_connect('key_press_event', key_press_event)
        plt.show()

    def fix_centers(self):
        fixed_locations = []
        fixed_locations_str = '['
        for clicked_location in self.centers:
            fixed_locations_str += '"(' + str(clicked_location[0]) + ', ' + str(clicked_location[1]) + ')",'

        fixed_locations_str = fixed_locations_str[:-1] + ']'

        print('"tagged_centers":', fixed_locations_str)

        fig, ax = plt.subplots()
        plt.imshow(self.img, cmap=plt.cm.gray)
        y = [center[0] for center in fixed_locations]
        x = [center[1] for center in fixed_locations]
        scatter_size = [3 for center in fixed_locations]
        plt.scatter(x, y, s=scatter_size)
        plt.show()


#     TODO: fix clicked centers
# Open last image - with alive points in blue (alive_centers = get_seen_centers_for_mask())
# When clicked - show red point + add to list
# When image closed - save list
# Enable revert

json_path = '5.3/exp_2024091102-02-1_type_5.3_mask_15_35_non-normalized_fixed.json'
# json_path = '5.3/exp_2024091002-05-3_type_5.3_formin_mask_15_35_non-normalized_fixed.json'
# json_path = '13.2/exp_2023071201-04-5_type_13.2_bleb_mask_15_35_non-normalized_fixed.json'
Labeler().label_video(json_path)
