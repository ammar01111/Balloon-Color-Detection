import cv2
import numpy as np

def detectBalloonColor(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(640,480))
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsv)

    colorRanges = {
        'red': [(0, 120, 70), (10, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'blue': [(100, 150, 0), (140, 255, 255)]
    }

    detectedColor = []
    
    for color,(upper,lower) in colorRanges.items():
        #upperBound = np.array(upper)
        #lowerBound = np.array(lower)

        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        mask = cv2.inRange(hsv,lower_red,upper_red)
        cv2.imshow('Masked', mask)

        contours , _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 500:

                image = cv2.drawContours(image,[cnt],-1,(0,255,0),-1)
                detectedColor.append(color)
    
    cv2.imshow('Detected Balloon', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    return detectedColor
imagePath = 'img1.jpg'
detectedColor = detectBalloonColor(imagePath)
print("Detected Balloon Color", detectedColor)


    
