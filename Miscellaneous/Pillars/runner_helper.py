from Pillars.analyzer import *


def get_circle_radius(config_data):
    exp_metadata = config_data["metadata"]
    micron = exp_metadata["micron"]
    micron_radius = exp_metadata["micron_radius"]
    pxl_radius = micron_radius / micron

    return pxl_radius
