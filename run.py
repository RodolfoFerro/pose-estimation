import cv2

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
    'THRESHOLD': 0.35,
    'SCALE': 2
}

# Initialize videocapture
CAPTURE = cv2.VideoCapture(0)

# Creates model instance
model = ModelLoader(MODEL_PATH)

host = '127.0.0.1'
port = 25001

# Creates viewer instance and runs capture
viewer = Viewer(model, CAPTURE, VIEWER_SPECS, host, port)
viewer.run()