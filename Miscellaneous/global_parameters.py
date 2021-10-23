import os

# todo: add explanation to all global parameters


SHOWFIG = True
SAVEFIG = not SHOWFIG

RECENT_DEATH_ONLY_FLAG = False

WINDOW_SIZE = 5

DIST_THRESHOLD_IN_PIXELS = 200

EPSILON = 1e-15

USE_LOG = False

CONFIG_FILES_DIR_PATH = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['config_files'])

ALL_EXPERIMENTS_FILES_MAIN_DIR = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV'])

METADATA_FILE_FULL_PATH = os.sep.join([ALL_EXPERIMENTS_FILES_MAIN_DIR, 'ExperimentsMetaData.csv'])

COMPRESSED_FILE_MAIN_DIR = os.sep.join([ALL_EXPERIMENTS_FILES_MAIN_DIR, 'CompressedTime_XYT_CSV'])

NON_COMPRESSED_FILE_MAIN_DIR = os.sep.join([ALL_EXPERIMENTS_FILES_MAIN_DIR, 'OriginalTimeMinutesData'])

LOWER_DEATH_PERCENTILE_BOUNDARY = 0.1
UPPER_DEATH_PERCENTILE_BOUNDARY = 0.9
