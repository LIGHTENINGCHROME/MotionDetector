import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Open the default camera
cam = cv2.VideoCapture(0)

isTrue, frame1 = cam.read()
isTrue, frame2 = cam.read()

li_contour=[]

while True:
    frame=cv2.absdiff(frame1,frame2)
    def contourfn(frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur =cv2.GaussianBlur(gray,(7,7),8)
        _, threshold=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
        dilated=cv2.dilate(threshold,None,iterations=3)
        contours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)   
    # Threshold of blue in HSV space 
    lower_blue = np.array([34, 110, 180]) 
    upper_blue = np.array([180, 255, 255]) 

    # preparing the mask to overlay 
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
        
    # The black region in the mask has the value of 0, 
    # so when multiplied with original image removes all non-blue regions 
    result = cv2.bitwise_and(frame1, frame1, mask = mask) 



    

    def contourIntersect(original_image, contour1, contour2):
        # Two separate contours trying to check intersection on
        contours = [contour1, contour2]

        # Create image filled with zeros the same size of original image
        blank = np.zeros(original_image.shape[0:2])

        # Copy each contour into its own image and fill it with '1'
        image1 = cv2.drawContours(blank.copy(), contours, 0, 1,thickness=cv2.FILLED)#filled contour  , 0 == element in list
        image2 = cv2.drawContours(blank.copy(), contours, 1, 1,thickness=cv2.FILLED)

        #cv2.imshow("camera3",image1)
        #cv2.imshow("camera4",image2)
        
        # Use the logical AND operation on the two images
        # Since the two images had bitwise AND applied to it,
        # there should be a '1' or 'True' where there was intersection
        # and a '0' or 'False' where it didnt intersect
        intersection = np.logical_and(image1, image2)
        
        # Check if there was a '1' in the intersection array
        return intersection.any()

    try:
        #for object
        c = max(contourfn(frame), key = cv2.contourArea)
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
        a=int(((x+w)+x)/2)
        b=int(((y+h)+y)/2)
        cv2.line(frame1,(a,y),(a,y+h),(255,0,0),2) #vertical
        cv2.line(frame1,(x,b),(x+w,b),(255,0,0),2) #horizontal
        cv2.circle(frame1,(a,b),10,(255,0,0),2)
        li_contour.append(c)

        #for color 
        c = max(contourfn(result), key = cv2.contourArea)
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
        a=int(((x+w)+x)/2)
        b=int(((y+h)+y)/2)
        cv2.circle(result,(a,b),10,(255,0,0),2)
        li_contour.append(c)
        print(contourIntersect(frame1, li_contour[0], li_contour[1]))#add motor control here 


    except ValueError as e:
        #print(e)
        cv2.imshow('Camera2', result)
        cv2.imshow('Camera', frame1)
        

    cv2.imshow('Camera2', result)
    cv2.imshow('Camera', frame1)
    frame1=frame2
    isTrue, frame2 = cam.read()

    li_contour.clear()
    time.sleep(.05)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
