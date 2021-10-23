import cv2
import numpy as np
from centroid import get_centroid
 
cap = cv2.VideoCapture("./video/video07.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

object_r = []
object_g = []
object_b = []

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    # 動きの少ないフレームは真っ黒にする
    if np.linalg.norm(rgb) < 20000 :
        rgb = np.zeros_like(rgb)
        for point in object_r:
            cv2.circle(frame2, point, 3, color=(0, 0, 255), thickness=-1)
        for point in object_g:
            cv2.circle(frame2, point, 3, color=(0, 255, 0), thickness=-1)
        for point in object_b:
            cv2.circle(frame2, point, 3, color=(255, 0, 0), thickness=-1)
        cv2.imshow('frame2',rgb)
        cv2.imshow('frame3',frame2)
        continue
    
    Z = rgb.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    # kmeans法でオプティカルフローのrgbを2色に分ける
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    rgb = res.reshape((rgb.shape))

    # 動きが激しいピクセルの座標の重心を求める(とても遅い)
    rcx, rcy, gcx, gcy, bcx, bcy, nonzero_r, nonzero_g ,nonzero_b = get_centroid(rgb)

    # リストに座標を記録する
    if nonzero_r != 0:
        object_r.append((rcx, rcy))
    if nonzero_g != 0:
        object_g.append((gcx, gcy))
    if nonzero_b != 0:
        object_b.append((bcx, bcy))
    
    # 重心の座標に赤い点をつける
    for point in object_r:
        cv2.circle(frame2, point, 3, color=(0, 0, 255), thickness=-1)
    for point in object_g:
        cv2.circle(frame2, point, 3, color=(0, 255, 0), thickness=-1)
    for point in object_b:
        cv2.circle(frame2, point, 3, color=(255, 0, 0), thickness=-1)
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