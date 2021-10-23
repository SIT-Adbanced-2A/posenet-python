import mymath.my_math as mm

import tensorflow as tf
import cv2
import time
import argparse
import numpy as np
import math

import posenet

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture("../video/video11.mp4")

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        cap.set(3, len(prvs))
        cap.set(4, len(prvs[0]))

        diameter = 100

        joint_mask = np.empty((diameter, diameter, 3), dtype=np.uint8)
        for y in range(diameter):
            for x in range(diameter):
                if np.linalg.norm((y - diameter / 2, x - diameter / 2)) <= diameter / 2:
                    joint_mask[y, x] = np.zeros(3, dtype=np.uint8)
                else:
                    joint_mask[y, x] = np.ones(3, dtype=np.uint8)

        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=0.7125, output_stride=output_stride)
            
            next = cv2.cvtColor(display_image.copy(), cv2.COLOR_BGR2GRAY)
            
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            for person in keypoint_coords:
                for joint in person:
                    joint = np.uint32(joint)
                    if np.all(person != 0):
                        min_y = int(joint[0] - diameter / 2) if int(joint[0] - diameter / 2) >= 0 else 0
                        max_y = int(joint[0] + diameter / 2) if int(joint[0] + diameter / 2) <= len(display_image) else len(display_image)
                        min_x = int(joint[1] - diameter / 2) if int(joint[1] - diameter / 2) >= 0 else 0
                        max_x = int(joint[1] + diameter / 2) if int(joint[1] + diameter / 2) <= len(display_image[0]) else len(display_image[0])
                        mask_min_y = min_y - int(joint[0] - diameter / 2)
                        mask_max_y = diameter - int(joint[0] + diameter / 2) + max_y
                        mask_min_x = min_x - int(joint[1] - diameter / 2)
                        mask_max_x = diameter - int(joint[1] + diameter / 2) + max_x
                        if min_y >= max_y or min_x >= max_x:
                            continue
                        display_image[min_y : max_y, min_x : max_x] *= joint_mask[mask_min_y : mask_max_y, mask_min_x : mask_max_x]

            display_image *= mm.create_mask(rgb.copy(), 90)
            for person in keypoint_coords:
                for joint in person:
                    joint = np.uint32(joint)
                    cv2.circle(display_image, (joint[1], joint[0]), 5, color=(255, 255, 255), thickness=1)
            prvs = next
            cv2.imshow('original', next)
            cv2.imshow('object_only', display_image)
            if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()