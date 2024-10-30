import cv2 
import numpy as np


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

    #for Red
    #contours, _ = cv2.findContours(maskRed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #for cnt in contours:
    #    x, y, w, h = cv2.boundingRect(cnt)
    #    cv2.rectangle(scaled,(x,y),((x+w),(y+h)),(0, 255, 255),2)
    
    #for Orange 
    contours, _ = cv2.findContours(maskOrange,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(scaled,(x,y),((x+w),(y+h)),(0, 165, 255),2)
    #for Pink 
    contours, _ = cv2.findContours(maskPink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(scaled,(x,y),((x+w),(y+h)),(0, 165, 255),2)


    cv2.imshow('Detected Orange Objects', outOrange)
    cv2.imshow('Detected Pink Objects', outPink)
    cv2.imshow('Bounding Boxes', scaled)



    if cv2.waitKey(1) == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()




