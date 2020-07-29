import cv2

from utils.model_loader import ModelLoader
from utils.viewer import viewer


WINDOW = 'Pose Estimation Demo'
MODEL_NAME = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
MODEL_PATH = 'models/' + MODEL_NAME

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2


# Creates model instance
model = ModelLoader(MODEL_PATH)

# Initialize videocapture:
cap = cv2.VideoCapture(0)
viewer(model.model_dict, cap, WINDOW)