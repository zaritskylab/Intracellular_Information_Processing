import os
import math


class Consts:
    __instance = None
    # Pillars consts that affects cache files
    RELATIVE_TO = 0.0519938
    # Mask
    SMALL_MASK_RADIUS = 15
    LARGE_MASK_RADIUS = 35

    normalized = False
    fixed = True
    inner_cell = True

    # Visualization consts
    only_alive = True
    build_image = True
    use_otsu = True
    pixel_to_whiten = 10
    gc_pvalue_threshold = 0.01
    percentage_from_perfect_circle_mask = 1
    MAX_DISTANCE_PILLAR_FIXED = 11
    MAX_DISTANCE_TO_CLOSEST = 5

    last_image_path = None
    fixed_images_path = None
    images_path = None

    consider_as_full_circle_percentage = 70

    pillar_to_intensities_cache_path = None
    correlation_alive_normalized_cache_path = None
    correlation_alive_not_normalized_cache_path = None
    all_pillars_correlation_normalized_cache_path = None
    all_pillars_correlation_not_normalized_cache_path = None
    frame2pillar_cache_path = None
    frame2alive_pillars_cache_path = None
    gc_df_cache_path = None
    alive_pillars_sym_corr_cache_path = None
    centers_cache_path = None
    pillar_to_neighbors_cache_path = None
    mask_for_each_pillar_cache_path = None
    gc_graph_cache_path = None
    last_img_alive_centers_cache_path = None
    IMAGE_SIZE = 1000
    CORRELATION = "pearson"
    FRAME_WINDOWS_AMOUNT = 10

    USE_CACHE = True
    RESULT_FOLDER_PATH = None
    BLOCK_TO_SHOW_GRAPH = True
    WRITE_OUTPUT = False


    # BFS data
    CIRCLE_RADIUS = 20
    MAX_CIRCLE_AREA = (math.pi * (CIRCLE_RADIUS + 3) ** 2) * 2
    CHECK_VALID_CENTER = 5
    CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH = CIRCLE_RADIUS - CHECK_VALID_CENTER
    CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH = CIRCLE_RADIUS + CHECK_VALID_CENTER
    # CIRCLE_RADIUS = 7
    # CHECK_VALID_CENTER = 2
    # CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH = CIRCLE_RADIUS - 2
    # CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH = CIRCLE_RADIUS + 2


    DIST_THRESHOLD_IN_PIXELS = 200

    CONFIG_FILES_DIR_PATH = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['config_files'])

    ALL_EXPERIMENTS_FILES_MAIN_DIR = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV'])

    METADATA_FILE_FULL_PATH = os.sep.join([ALL_EXPERIMENTS_FILES_MAIN_DIR, 'ExperimentsMetaData.csv'])

    COMPRESSED_FILE_MAIN_DIR = os.sep.join([ALL_EXPERIMENTS_FILES_MAIN_DIR, 'CompressedTime_XYT_CSV'])

    NON_COMPRESSED_FILE_MAIN_DIR = os.sep.join([ALL_EXPERIMENTS_FILES_MAIN_DIR, 'OriginalTimeMinutesData'])

    ALL_TREATMENT_EXPERIMENTS_DIR = os.sep.join([ALL_EXPERIMENTS_FILES_MAIN_DIR, 'AllTreatmentExperiments'])

    PILLARS = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data', 'Pillars'])

    FIXED_IMAGES = os.sep.join([PILLARS, 'FixedImages'])

    PILLAR_PERCENTAGE_MUST_PASS = 10

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Consts.__instance == None:
            Consts()
        return Consts.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Consts.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Consts.__instance = self
