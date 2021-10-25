import mymath.my_math as mm

import tensorflow as tf
import cv2
import time
import sys
import numpy as np
import math

import posenet

def main():
    with tf.compat.v1.Session() as sess:
        args = sys.argv
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture(args[1])

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        cap.set(3, len(prvs))
        cap.set(4, len(prvs[0]))

        frame_count = 0

        diameter = 140

        record_size = 100
        recording = False

        right_sholder_angle = 0
        left_sholder_angle = 0

        writer = None

        # 直近5フレーム内の放り投げ容疑の度合いを持つ配列
        suspection_points = np.zeros(5, dtype=np.uint8)
        # 直近100フレームを持つ配列
        old_frames = np.zeros((record_size, len(prvs), len(prvs[0]), 3), dtype=np.uint8)
        # 関節周りの画素にマスクする配列(直径の幅がdiameterの円)
        joint_mask = np.empty((diameter, diameter, 3), dtype=np.uint8)
        for y in range(diameter):
            for x in range(diameter):
                if np.linalg.norm((y - diameter / 2, x - diameter / 2)) <= diameter / 2:
                    joint_mask[y, x] = np.zeros(3, dtype=np.uint8)
                else:
                    joint_mask[y, x] = np.ones(3, dtype=np.uint8)

        while True:
            try:
                input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=0.7125, output_stride=output_stride)
                
                next = cv2.cvtColor(display_image.copy(), cv2.COLOR_BGR2GRAY)
                old_frames[frame_count % record_size] = display_image.copy()
                
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

                # 関節周りの画素を黒塗りする
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

                # オプティカルフローのマスクを掛ける
                display_image *= mm.create_mask(rgb.copy(), 150)
                # 動いている物体の重心を取得する
                cnt, nonzero = mm.get_rgb_array_centroid(display_image)
                curr_right_sholder_angle = 0
                curr_left_sholder_angle = 0
                curr_right_waist_angle = 0
                curr_left_waist_angle = 0
                sholder_angv = 0
                right_hand_distance = 0
                left_hand_distance = 0
                # 関節の角度や角速度、手から荷物の距離を計測する
                for person in keypoint_coords:
                    if np.all(person != 0):
                        curr_right_sholder_angle = mm.get_angle(person[7] - person[5], person[11] - person[5])
                        curr_left_sholder_angle = mm.get_angle(person[8] - person[5], person[12] - person[5])
                        curr_right_waist_angle = mm.get_angle(person[5] - person[11], person[13] - person[11])
                        curr_left_waist_angle = mm.get_angle(person[6] - person[12], person[14] - person[12])
                        sholder_angv = curr_right_sholder_angle - right_sholder_angle
                        left_hand_distance = np.linalg.norm(person[9] - cnt)
                        right_hand_distance = np.linalg.norm(person[10] - cnt)
                        if abs(curr_left_sholder_angle - left_sholder_angle) > abs(sholder_angv):
                            sholder_angv = curr_left_sholder_angle - left_sholder_angle
                        right_sholder_angle = curr_right_sholder_angle
                        left_sholder_angle = curr_left_sholder_angle
                        for joint in person:
                            joint = np.uint32(joint)
                            cv2.circle(display_image, (joint[1], joint[0]), 5, color=(255, 255, 255), thickness=1)

                # オプティカルフローで検出された画素が1つでもある場合は容疑度合いを計算する
                if nonzero != 0:
                    distance = right_hand_distance if right_hand_distance > left_hand_distance else left_hand_distance
                    if distance < 100:
                        distance = 100
                    waist_angle = curr_right_waist_angle if abs(curr_right_waist_angle) > abs(curr_left_waist_angle) else curr_left_waist_angle
                    # 現在のフレームにおける容疑の度合いを計算する(容疑の度合いが強いとobject_onlyのフレームの荷物が赤くマークされる)
                    red = pow(sholder_angv, 2) * nonzero * pow(distance - 100, 2) / (10000000 * abs(waist_angle - 180) + 1) * 255
                    if red > 255:
                        red = 255
                    red = int(red)
                    suspection_points[frame_count % len(suspection_points)] = red
                    # 5フレーム内のマークの色の赤成分の平均が200を超えている場合はレコーディングを開始する
                    if np.sum(suspection_points) > 200 * 5 and not(recording):
                        recording = True
                        video_format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                        writer = cv2.VideoWriter("./" + str(frame_count) + ".mp4", video_format, 60, (len(display_image[0]), len(display_image)))
                        # 99フレーム前から直近のフレームまでファイルに書き出す
                        for i in range(1, 100):
                            writer.write(old_frames[(frame_count + i) % 100])

                    # レコーディングが有効な場合、現在のフレームをファイルに書き出す
                    if recording:
                        writer.write(old_frames[frame_count % 100])
                    
                    # 動きの少ないフレームが連続した場合、レコーディングを終了する
                    if np.sum(suspection_points) < 200 and recording:
                        recording = False
                        writer.release()
                        writer = None
                    # 荷物にマーカーを付ける
                    cv2.circle(display_image, (cnt[1], cnt[0]), 50, color=(255 - red, 255 - red, red), thickness=1)
                    cv2.line(display_image, (cnt[1] - 60, cnt[0]), (cnt[1] - 40, cnt[0]), color=(255 - red, 255 - red, red), thickness=1)
                    cv2.line(display_image, (cnt[1] + 40, cnt[0]), (cnt[1] + 60, cnt[0]), color=(255 - red, 255 - red, red), thickness=1)
                    cv2.line(display_image, (cnt[1], cnt[0] - 60), (cnt[1], cnt[0] - 40), color=(255 - red, 255 - red, red), thickness=1)
                    cv2.line(display_image, (cnt[1], cnt[0] + 40), (cnt[1], cnt[0] + 60), color=(255 - red, 255 - red, red), thickness=1)
                else:
                    # 容疑無し
                    suspection_points[frame_count % len(suspection_points)] = 0

                prvs = next
                cv2.imshow('current_frame', old_frames[frame_count % 100])
                cv2.imshow('object_only', display_image)
                cv2.imshow('-99 frame', old_frames[(frame_count + 1) % 100])
                frame_count += 1
                if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                    if writer != None:
                    # ファイルに書き出し中の場合はファイルを閉じる
                        writer.release()
                    break
            
            except:
                # エラー処理
                if writer != None:
                    # ファイルに書き出し中の場合はファイルを閉じる
                    writer.release()
                    break
        cap.release()
        print("VideoCapture was closed")
        cv2.destroyAllWindows()
        print("Windows were closed")
        sess.close()
        print("Session was closed")

if __name__ == "__main__":
    main()