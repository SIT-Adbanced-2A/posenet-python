import cv2
import numpy as np
cap = cv2.VideoCapture("./video/video02.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    #y = 0
    #for row in rgb :
    #print("height : " + str(y))
    #    y += 1
    #    x = 0
    #    for pixel in row :
    #        if np.linalg.norm(pixel) > np.linalg.norm([60, 60, 60]):
    #            print("(x, y) = ({}, {})".format(x, y))
    #            print(pixel)
    #        x += 1

    # 動きが少ないフレームは画面を真っ黒にする
    #if np.linalg.norm(rgb) < 20000 :
    #    rgb = np.zeros_like(rgb)
        
    # 動きの激しさに色を付けて画面に映す
    cv2.imshow('frame2',rgb)
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