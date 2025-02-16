import cv2
import numpy as np


# Open the default camera
cam = cv2.VideoCapture(0)

isTrue, frame1 = cam.read()
isTrue, frame2 = cam.read()

while True:
    frame=cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur =cv2.GaussianBlur(gray,(7,7),8)
    _, threshold=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(threshold,None,iterations=3)
    contours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    try:
        c = max(contours, key = cv2.contourArea)
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
        a=int(((x+w)+x)/2)
        b=int(((y+h)+y)/2)
        cv2.line(frame1,(a,y),(a,y+h),(255,0,0),2)
        cv2.line(frame1,(x,b),(x+w,b),(255,0,0),2)
    except ValueError:
        cv2.imshow('Camera', frame1)
        


    cv2.imshow('Camera', frame1)
    frame1=frame2
    isTrue, frame2 = cam.read()

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
