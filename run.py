import cv2
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')

from utils.model_loader import ModelLoader
from utils.viewer import Viewer


MODEL_NAME = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
MODEL_PATH = 'models/' + MODEL_NAME

VIEWER_SPECS = {
    'WINDOW_NAME': 'Pose Estimation',
    'MIRROR_IMAGE': True,
    'DRAW_LINKS': True,
    'POINT_COLOR': (156, 245, 66),
    'LINK_COLOR': (255, 0, 76),
    'THICKNESS': 1,
    'THRESHOLD': 0.85,
    'SCALE': 2
}

OUT_FILE = 'out.json'


# Initialize videocapture
CAPTURE = cv2.VideoCapture(0)

# Creates model instance
model = ModelLoader(MODEL_PATH)

# Creates viewer instance and runs capture
viewer = Viewer(model, CAPTURE, VIEWER_SPECS, OUT_FILE)
viewer.run()