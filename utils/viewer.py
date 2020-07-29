from time import time

import numpy as np
import cv2


def parse_output(heatmap_data,offset_data, threshold):
    '''
    Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
    Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
    '''

    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num,3), np.uint32)

    for i in range(heatmap_data.shape[-1]):
        joint_heatmap = heatmap_data[...,i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
        pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
        pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
        max_prob = np.max(joint_heatmap)

        if max_prob > threshold:
            if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
                pose_kps[i,2] = 1

    return pose_kps


def draw_kps(show_img,kps, ratio=None):
    for i in range(5,kps.shape[0]):
        if kps[i,2]:
            if isinstance(ratio, tuple):
                cv2.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
                continue
            cv2.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img


def abs_compare(keypoints, pair, eps_max):
    if np.abs(keypoints[pair[0]][1]) < eps_max and np.abs(keypoints[pair[0]][0]) < eps_max and np.abs(keypoints[pair[1]][1]) < eps_max and np.abs(keypoints[pair[1]][0]) < eps_max:
        return True
    else:
        return False


def draw_body(img, keypoints, pairs):
    color = (0,255,0)
    eps_max = 200
    for i, pair in enumerate(pairs):    
        if abs_compare(keypoints, pair, eps_max):
            cv2.line(img, (keypoints[pair[0]][1], keypoints[pair[0]][0]), (keypoints[pair[1]][1], keypoints[pair[1]][0]), color=color, lineType=cv2.LINE_AA, thickness=1)
    
    return img


def viewer(model, capture, window):
    """FER viewer."""

    interpreter = model['interpreter']
    input_details = model['input_details']
    output_details = model['output_details']

    parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

    while True:
        # Capture frame-by-frame:
        ret, frame = capture.read()

        # Extract input details from loaded model
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]

        # Prepare input image
        in_img = cv2.resize(frame[:, 280:-280], (width, height))
        in_img = np.expand_dims(in_img, axis=0)

        float_model = input_details[0]['dtype'] == np.float32
        if float_model:
            in_img = (np.float32(in_img) - 127.5) / 127.5

        # Set the value of the input tensor
        interpreter.set_tensor(input_details[0]['index'], in_img)

        # Start time count
        start_prediction = time()

        # Run inference from model
        interpreter.invoke()  # HERE COMES THE MAGIC!

        # Extract output data from the interpreter.
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        offset_data = interpreter.get_tensor(output_details[1] ['index'])

        # Parse output
        heatmaps = np.squeeze(output_data)
        offsets = np.squeeze(offset_data)
        kps = parse_output(heatmaps, offsets, 0.7)

        # Process and draw output
        out_img = np.squeeze((in_img.copy() * 127.5 + 127.5) / 255.)
        out_img = np.array(out_img * 255, np.uint8)
        
        # out_img = draw_body(out_img, kps, parts_to_compare)
        out_img = draw_kps(out_img, kps)

        # End time count
        end_prediction = time()
        delta_prediction = end_prediction - start_prediction
        print(" * Time for prediction: {}".format(delta_prediction))

        # Display the resulting frame
        cv2.imshow(window, out_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()