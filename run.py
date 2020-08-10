import cv2

from utils.model_loader import ModelLoader
from utils.viewer import Viewer


MODEL_NAME = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
MODEL_PATH = 'models/' + MODEL_NAME

VIEWER_SPECS = {
    'WINDOW_NAME': 'Pose Estimation',
    'POINT_COLOR': (66, 245, 156),
    'LINK_COLOR': (66, 185, 245),
    'THICKNESS': 1
}

OUT_FILE = 'out.json'


# Initialize videocapture
CAPTURE = cv2.VideoCapture(0)

# Creates model instance
model = ModelLoader(MODEL_PATH)

# Creates viewer instance and runs capture
viewer = Viewer(model, CAPTURE, VIEWER_SPECS, OUT_FILE)
viewer.run()