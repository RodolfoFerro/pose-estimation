from multiprocessing import Process


def process_viewer(camera):
    """Processes a viewer instance as a subprocess.

    Parameters
    ----------
    camera : int
        An integer specifying the camera ID to be loaded.
    """

    import psutil
    import cv2

    from utils.model_loader import ModelLoader
    from utils.viewer import Viewer


    mapper = {
        0: 'A',
        1: 'B'
    }

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

    # Specify output file
    OUT_FILE = f'out_{mapper[camera]}.json'

    # Psutil Process (CPU affinity)
    p = psutil.Process()
    p.cpu_affinity([camera + 1])

    # Initialize videocapture
    CAPTURE = cv2.VideoCapture(camera)

    # Creates model instance
    model = ModelLoader(MODEL_PATH)

    # Creates viewer instance and runs capture
    viewer = Viewer(model, CAPTURE, VIEWER_SPECS, OUT_FILE)
    viewer.run()


if __name__ == "__main__":
    camera = -1
    procs = 2
    jobs = []

    for i in range(procs):
        camera += 1
        process = Process(target=process_viewer, args=([camera]))
        jobs.append(process)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()
