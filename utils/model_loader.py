import tensorflow as tf


class ModelLoader():

    def __init__(self, model_path):
        """Creates a tflite model instance."""

        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.model_dict = dict(
            interpreter=interpreter,
            input_details=input_details,
            output_details=output_details
        )

        print("[INFO] The model has been loaded successfully!")