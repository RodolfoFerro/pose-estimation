from multiprocessing import Process

import cv2

from utils.model_loader import ModelLoader
from utils.viewer import Viewer

from utils.multicam import process_viewer


MODEL_NAME = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
MODEL_PATH = 'models/' + MODEL_NAME

VIEWER_SPECS = {
    'WINDOW_NAME': 'Pose Estimation',
    'MIRROR_IMAGE': True,
    'POINT_COLOR': (66, 245, 156),
    'LINK_COLOR': (66, 185, 245),
    'THICKNESS': 1,
    'THRESHOLD': 0.85
}

OUT_FILE_A = 'out_A.json'
OUT_FILE_B = 'out_B.json'


# Initialize videocapture
CAPTURE_A = cv2.VideoCapture(0)
CAPTURE_B = cv2.VideoCapture(1)

# Creates model instance
model_A = ModelLoader(MODEL_PATH)
model_B = ModelLoader(MODEL_PATH)

# Creates viewer instance and runs capture
viewer_A = Viewer(model_A, CAPTURE_A, VIEWER_SPECS, OUT_FILE_A)
viewer_B = Viewer(model_B, CAPTURE_B, VIEWER_SPECS, OUT_FILE_B)
viewers = [viewer_A, viewer_B]


if __name__ == "__main__":
    pid = -1
    jobs = []

    for viewer in viewers:
        pid += 1
        process = Process(target=process_viewer, args=(viewer, pid))
        jobs.append(process)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()