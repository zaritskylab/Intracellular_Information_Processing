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
from Pillars.video_cropper import *

import json
import math


class Cropper:
    def __init__(self):
        self.centers = []
        self.lns = []
        self.exp = '20230818'
        self.video = '04'
        self.cell_index = 1

    def crop_cells(self, tif_path):
        full_img_stack = io.imread(tif_path)
        self.img = full_img_stack[len(full_img_stack) - 1]
        self.start()

    def start(self):
        def onclick(event):
            x = int(event.ydata)
            y = int(event.xdata)

            clicked_location = (x, y)

            self.lns.append(
                plt.plot([clicked_location[1]], [clicked_location[0]], ls='none', marker='.', ms=5, c='red'))
            self.centers.append(clicked_location)
            plt.show()

        def key_press_event(event):
            if event.key == 'z':
                if len(self.centers) > 0:
                    self.centers.pop()
                    last_added_point = self.lns.pop()
                    last_added_point = last_added_point.pop()
                    if last_added_point is not None:
                        last_added_point.remove()
                        plt.show()
            elif event.key == 'c':
                print("crop")
                save_path = get_save_path(self.exp, self.video, self.cell_index)
                crop_video_rect(tif_file,
                                self.centers[0], self.centers[1],
                                save_path)
                self.cell_index += 1

                while len(self.centers) > 0:
                    self.centers.pop()
                    last_added_point = self.lns.pop()
                    last_added_point = last_added_point.pop()
                    if last_added_point is not None:
                        last_added_point.remove()
                plt.show()

        def get_save_path(exp, video, cell_num):
            return 'C:\\Users\\Sarit Hollander\\Desktop\\Study\\MSc\\Research\\Project\\Intracellular_Information_Processing\\Data\\Pillars\\5.3\\exp-' + exp + '-video-' + video + '-cell-' + str(cell_num) + '-Airyscan Processing.tif'

        fig, ax = plt.subplots()
        plt.imshow(self.img, cmap=plt.cm.gray)
        # y = [center[0] for center in self.alive_centers]
        # x = [center[1] for center in self.alive_centers]
        # scatter_size = [3 for center in self.alive_centers]
        # plt.scatter(x, y, s=scatter_size)
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', key_press_event)
        plt.show()

    # def show_centers(self):
    #     fig, ax = plt.subplots()
    #     fig, ax = plt.subplots()
    #     plt.imshow(self.img, cmap=plt.cm.gray)
    #     y = [center[0] for center in fixed_locations]
    #     x = [center[1] for center in fixed_locations]
    #     scatter_size = [3 for center in fixed_locations]
    #     plt.scatter(x, y, s=scatter_size)
    #     plt.show()


tif_file = 'C:\\Users\\Sarit Hollander\\Desktop\\Study\\MSc\\Research\\Project\\Intracellular_Information_Processing\\Data\\Pillars\\5.3\\20230818-video-04-Airyscan Processing.tif'
Cropper().crop_cells(tif_file)
