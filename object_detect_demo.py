import cv2
import numpy as np
from mask import create_mask_cy
 
cap = cv2.VideoCapture("./video/video11.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

diameter = 100

joint_mask = np.empty((diameter, diameter, 3), dtype=np.uint8)
for y in range(diameter):
    for x in range(diameter):
        if np.linalg.norm((y - diameter / 2, x - diameter / 2)) <= diameter / 2:
            joint_mask[y, x] = np.zeros(3, dtype=np.uint8)
        else:
            joint_mask[y, x] = np.ones(3, dtype=np.uint8)

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    joints = np.array([[0, 0], [100, 100], [200, 200], [300, 300], [400, 400], [500, 500], [600, 600]], dtype=np.int32)

    mask = create_mask_cy(rgb.copy(), 90)
    for j in joints:
        min_y = int(j[0] - diameter / 2) if int(j[0] - diameter / 2) >= 0 else 0
        max_y = int(j[0] + diameter / 2) if int(j[0] + diameter / 2) <= len(frame2) else len(frame2)
        min_x = int(j[1] - diameter / 2) if int(j[1] - diameter / 2) >= 0 else 0
        max_x = int(j[1] + diameter / 2) if int(j[1] + diameter / 2) <= len(frame2[0]) else len(frame2[0])
        if min_y >= max_y or min_x >= max_x:
            continue
        frame2[min_y : max_y, min_x : max_x] *= joint_mask[min_y - int(j[0] - diameter / 2) : diameter - int(j[0] + diameter / 2) + max_y, min_x - int(j[1] - diameter / 2) : diameter - int(j[1] + diameter / 2) + max_x]
    frame2 *= mask
    for p in joints:
        cv2.circle(frame2, (p[0], p[1]), 5, color=(255, 255, 255), thickness=-1)
    
    # 動きの激しさに色を付けて画面に映す
    cv2.imshow('frame2',rgb)
    cv2.imshow('frame3',frame2)
    # ESCキーで処理を中断する
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()