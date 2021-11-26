from time import time
import json

import numpy as np
import cv2


class Viewer():

    def __init__(self, model, video_capture, viewer_specs, output_file=None,
        socket_conn=False):
        """Loads model, video capture and viewer.

        Sets interpreter and viewer specifications.

        Parameters
        ----------
        model : model dict
            Dictionary containing interpreter, input and output details
            of the model to be used for inference.
        video_capture : cv2 capture
            Specifies the capture (input) to be used for video input.
        viewer_specs : dict
            Dicitonary containing extra parameters for viewing inference
            outputs.
        """

        self.interpreter = model.model_dict['interpreter']
        self.input_details = model.model_dict['input_details']
        self.output_details = model.model_dict['output_details']
        self.capture = video_capture
        self.socket_conn = socket_conn

        specs = viewer_specs.keys()

        self.window = viewer_specs['WINDOW_NAME'] \
            if 'WINDOW_NAME' in specs else 'CAPTURE'
        self.mirror = viewer_specs['MIRROR_IMAGE'] \
            if 'MIRROR_IMAGE' in specs else False
        self.links = viewer_specs['DRAW_LINKS'] \
            if 'DRAW_LINKS' in specs else False
        self.point_color = viewer_specs['POINT_COLOR'] \
            if 'POINT_COLOR' in specs else (255, 0, 0)
        self.link_color = viewer_specs['LINK_COLOR'] \
            if 'LINK_COLOR' in specs else (255, 255, 0)
        self.thickness = viewer_specs['THICKNESS'] \
            if 'THICKNESS' in specs else 1
        self.threshold = viewer_specs['THRESHOLD'] \
            if 'THRESHOLD' in specs else 0.7
        self.scale = viewer_specs['SCALE'] \
            if 'SCALE' in specs else 1
        self.squared = viewer_specs['SQUARED'] \
            if 'SQUARED' in specs else False
        
        self.output_file = output_file


    def _parse_output(self, heatmap_data, offset_data):
        '''Parses output from inference.

        Parameters
        ----------
        heatmap_data : ndarray (3-dimensional)
            The heatmaps for an image. These vectors are obtained from
            inference.
        offset_data : ndarray (3-dimensional)
            The offset vectors for an image. These vectors are obtained
            from inference.

        Returns
        -------
        pose_kps : ndarray
            Contains the (x, y) paired values of the keypoints and flags for
            points with low probabilities.
        '''

        joint_num = heatmap_data.shape[-1]
        pose_kps = np.zeros((joint_num, 3), np.uint32)

        for i in range(heatmap_data.shape[-1]):
            # Compute max ocurrences
            joint_heatmap = heatmap_data[..., i]
            max_coincidences = joint_heatmap == np.max(joint_heatmap)
            max_val_pos = np.squeeze(np.argwhere(max_coincidences))
            remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.float64)

            # Compute (x, y) positions
            x, y = remap_pos[0], remap_pos[1]
            x += offset_data[max_val_pos[0], max_val_pos[1], i]
            y += offset_data[max_val_pos[0], max_val_pos[1], i + joint_num]
            pose_kps[i, 0] = int(x)
            pose_kps[i, 1] = int(y)
            max_prob = np.max(joint_heatmap)

            # Assign values or non-prob flag
            if max_prob > self.threshold:
                if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
                    pose_kps[i,2] = 1

        return pose_kps


    def _serialize_output(self, kps):
        """Serializes output.

        Writes output keypoints into a JSON file.
        """

        json_output = []
        parts = [
            'NOSE', 'L_EYE', 'R_EYE', 'L_EAR', 'R_EAR', 'L_SHOULDER',
            'R_SHOULDER', 'L_ELBOW', 'R_ELBOW', 'L_WRIST', 'R_WRIST',
            'L_HIP', 'R_HIP', 'L_KNEE', 'R_KNEE', 'L_ANKLE', 'R_ANKLE'
        ]
        for i in range(kps.shape[0]):
            if kps[i, 2]:
                part = {
                    'ID': i,
                    'part': parts[i],
                    'x': int(kps[i, 1]),
                    'y': int(kps[i, 0])
                }
                json_output.append(part)
        
        output = json.dumps(json_output, indent=4)

        # with open(self.output_file, 'w') as f:
        #     f.write(output)


    def _serialize_to_socket(self, kps, connection):
        """Serializes output to socket.

        Writes output keypoints into a JSON file.
        """

        parts = [
            'NOSE', 'L_EYE', 'R_EYE', 'L_EAR', 'R_EAR', 'L_SHOULDER',
            'R_SHOULDER', 'L_ELBOW', 'R_ELBOW', 'L_WRIST', 'R_WRIST',
            'L_HIP', 'R_HIP', 'L_KNEE', 'R_KNEE', 'L_ANKLE', 'R_ANKLE'
        ]
        json_msg = ''

        for i in range(kps.shape[0]):
            if kps[i, 2]:
                part = {
                    'ID': i,
                    'part': parts[i],
                    'x': int(kps[i, 1]),
                    'y': int(kps[i, 0])
                }
                json_msg += json.dumps(part) + ';'
        
        json_msg = json_msg[:-1]
        print(json_msg)
        connection.send(json_msg.enconde())
    

    def _draw_links(self, show_img, kps, ratio=None):
        """Draws a link from position A to position B.
        
        Parameters
        ----------
        show_img : ndarray
            The image where the keypoints will be plotted.
        kps : ndarray
            The tensor containing the keypoints to be plotted.
        ratio : float
            A ratio to scale the plotting points in output image.

        Returns
        -------
        show_img : ndarray
            The output image to be shown, containing the plotted link.
        """
        
        valid_links = [
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        for link in valid_links:
            i, j = link
            if kps[i, 2] and kps[j, 2]:
                if isinstance(ratio, tuple):
                    a_X = int(round(kps[i, 1] * ratio[1]))
                    a_Y = int(round(kps[i, 0] * ratio[0]))
                    b_X = int(round(kps[j, 1] * ratio[1]))
                    b_Y = int(round(kps[j, 0] * ratio[0]))

                    cv2.line(
                        show_img,
                        (a_X, a_Y),
                        (b_X, b_Y),
                        self.link_color,
                        self.thickness
                    )
                    continue

                cv2.line(
                    show_img,
                    (kps[i, 1], kps[i, 0]),
                    (kps[j, 1], kps[j, 0]),
                    self.link_color,
                    self.thickness
                )
        
        return show_img


    def _draw_kps(self, show_img, kps, ratio=None):
        """Processes all keypoints and plots them in output image.

        Parameters
        ----------
        show_img : ndarray
            The image where the keypoints will be plotted.
        kps : ndarray
            The tensor containing the keypoints to be plotted.
        ratio : float
            A ratio to scale the plotting points in output image.

        Returns
        -------
        show_img : ndarray
            The output image to be shown, containing the plotted keypoints.
        """

        if self.links:
            show_img = self._draw_links(
                show_img, 
                kps
            )

        for i in range(kps.shape[0]):
            if kps[i, 2]:
                if isinstance(ratio, tuple):
                    p_X = int(round(kps[i, 1] * ratio[1]))
                    p_Y = int(round(kps[i, 0] * ratio[0]))

                    cv2.circle(
                        show_img,
                        (p_X, p_Y),
                        self.thickness,
                        self.point_color,
                        round(int(1 * ratio[1]))
                    )
                    continue

                cv2.circle(
                    show_img,
                    (kps[i, 1], kps[i, 0]),
                    self.thickness,
                    self.point_color,
                    self.thickness
                )

        return show_img


    def run(self, connection=None):
        """Runs viewer.

        Function to invoke interpreter, run inference and plot results.
        """
        while True:
            # Capture frame-by-frame:
            ret, frame = self.capture.read()
            fh, fw = frame.shape[:2]
            delta = (fw - fh) // 2

            # Extract input details from loaded model
            input_shape = self.input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]

            # Prepare input image
            if self.mirror:
                frame = cv2.flip(frame, 1)

            in_img = frame[:, delta:-delta]

            in_img = cv2.resize(in_img, (width, height))
            in_img = np.expand_dims(in_img, axis=0)

            float_model = self.input_details[0]['dtype'] == np.float32
            if float_model:
                in_img = (np.float32(in_img) - 127.5) / 127.5

            # Set the value of the input tensor
            in_details = self.input_details[0]['index']
            self.interpreter.set_tensor(in_details, in_img)

            # Start time count
            start_prediction = time()

            # Run inference from model
            self.interpreter.invoke()  # HERE COMES THE MAGIC!

            # Extract output data from the interpreter.
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            out_details = self.output_details[0]['index']
            off_details = self.output_details[1]['index']
            output_data = self.interpreter.get_tensor(out_details)
            offset_data = self.interpreter.get_tensor(off_details)

            # Parse output
            heatmaps = np.squeeze(output_data)
            offsets = np.squeeze(offset_data)
            kps = self._parse_output(heatmaps, offsets)

            # Process and draw output
            out_img = np.squeeze((in_img.copy() * 127.5 + 127.5) / 255.)
            out_img = np.array(out_img * 255, np.uint8)
            out_img = self._draw_kps(out_img, kps)

            # Scale output
            oh, ow = out_img.shape[:2]
            scale_size = (int(ow * self.scale), int(oh * self.scale))
            out_img = cv2.resize(out_img, scale_size)

            # Writes output file
            if self.output_file != None:
                self._serialize_output(kps)

            if self.socket_conn and connection:
                self._serialize_to_socket(kps, connection)

            # End time count
            end_prediction = time()
            delta_prediction = end_prediction - start_prediction
            # print(" * Time for prediction: {}".format(delta_prediction))

            # Display the resulting frame
            cv2.imshow(self.window, out_img)
            # cv2.imshow(self.window, new_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()
