import cv2 
import numpy as np

scale = .75
feed = cv2.VideoCapture(0)

while True:

    success, image = cv2.imread(feed)

    scaled = cv2.resize(image,(640,480), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(scaled,cv2.COLOR_BGR2HSV)

    #lower_red = np.array([0, 120, 70])
    #upper_red = np.array([10, 255, 255])

    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])

    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([35, 255, 255])

    #maskRed = cv2.inRange(hsv,lower_red,upper_red)
    maskOrange = cv2.inRange(hsv,lower_orange,upper_orange)
    maskPink = cv2.inRange(hsv,lower_pink,upper_pink)

    #outRed = cv2.bitwise_and(scaled,scaled,mask=maskRed)
    outOrange = cv2.bitwise_and(scaled,scaled,mask=maskOrange)
    outPink = cv2.bitwise_and(scaled,scaled,mask=maskPink)


    #cv2.imshow('outRed',outRed)
    cv2.imshow('outOrange',outOrange)
    cv2.imshow('outPink',outPink)


    if cv2.waitKey(1) == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()




