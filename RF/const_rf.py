from os.path import join, abspath, dirname, pardir
BASE_DIR = abspath(join(dirname(__file__), pardir))
output_dir = join(BASE_DIR, 'RF/dataset/')
split_mark = '\t'
OPEN_WORLD = False
MONITORED_SITE_NUM = 4
MONITORED_INST_NUM = 100
UNMONITORED_SITE_NUM = 40000
UNMONITORED_SITE_TRAINING = 900
model_path = 'pretrained/'

num_classes = 4
num_classes_ow = 0
# Length of TAM
max_matrix_len = 18000
# Maximum Load Time
maximum_load_time = 800

max_trace_length = 100000
