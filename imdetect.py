import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Open the default camera
cam = cv2.VideoCapture(0)#start video capture

isTrue, frame1 = cam.read()#take 1st frame
isTrue, frame2 = cam.read()#take 2nd frame


#frame normalization
frame_1=cv2.normalize(frame1,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)#normalized frame1
frame_2=cv2.normalize(frame2,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)#normalized frame2



li_contour=[]#empty contour list for contours from color detector and motion detector for comparison and to find intersection



while True:
    frame=cv2.absdiff(frame_1,frame_2)#find the difference
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#gray image to match the shape of mask

    #contour detection function
    def contourfn(frame,bool):
        if bool:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert to grayscale if bool == True
        else:
            gray=frame
        blur =cv2.GaussianBlur(gray,(7,7),8) #apply blur to find the major contours
        _, threshold=cv2.threshold(blur,20,255,cv2.THRESH_BINARY) #map out the general shape 
        dilated=cv2.dilate(threshold,None,iterations=3) #increase the sizeof the boundarys found
        contours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find the countours
        return contours #return countours

    #################################################################################################
    #frame processing of frame1 for color detection
    #################################################################################################
    hsv_intial= cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV) #colorspace convertion
    hsv = cv2.normalize(hsv_intial,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#normalization

    # Threshold of blue in HSV space 
    lower_blue = np.array([34, 110, 180]) 
    upper_blue = np.array([180, 255, 255]) 

    # preparing the mask to overlay 
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
        
    # The black region in the mask has the value of 0, 
    # so when multiplied with original image removes all non-blue regions 
    result = cv2.bitwise_and(frame1, frame1, mask = mask) 

    #################################################################################################
    #contour intersection function
    #################################################################################################

    def contourIntersect(original_image, contour1, contour2):
        # Two separate contours trying to check intersection on
        contours = [contour1, contour2]

        # Create image filled with zeros the same size of original image
        blank = np.zeros(original_image.shape[0:2])

        # Copy each contour into its own image and fill it with '1'
        image1 = cv2.drawContours(blank.copy(), contours, 0, 1,thickness=cv2.FILLED)#filled contour  , 0 == element in list
        image2 = cv2.drawContours(blank.copy(), contours, 1, 1,thickness=cv2.FILLED)#filled contour  , 1 == element in list

        
        # Use the logical AND operation on the two images
        # Since the two images had bitwise AND applied to it,
        # there should be a '1' or 'True' where there was intersection
        # and a '0' or 'False' where it didnt intersect
        intersection = np.logical_and(image1, image2)
        
        # Check if there was a '1' in the intersection array
        return intersection.any()
    
    ###############################################################################################################################
    #face detection using haars cascade 
    ###############################################################################################################################


    image = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #convert the image to grayscale for face detection
    image_normalized = cv2.normalize(image,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX) #normalize the image
    
    harrs=cv2.CascadeClassifier("haar_faceside.xml")#import haar cascade

    faces=harrs.detectMultiScale(image_normalized,scaleFactor=1.1,minNeighbors=3)#define the parameters

    blank = np.zeros(image_normalized.shape[:2])#make a blank image with same size as reference image

    for (x,y,w,h) in faces:
    
        detect_mask=cv2.rectangle(blank,(x,y),(x+w,y+h),(255,255,255),thickness=-1)#map the rectangular coordinates on blank image
                                                                                    #and fill the whole thing to make to  make a mask

    cv2.imshow("body",blank)

    ################################################################################################################################
    #overlay drawing accross countours detected
    ################################################################################################################################


    try:
        mask = blank.astype("uint8")#convert mask to uint8
        masked_frame=cv2.bitwise_and(gray_frame,gray_frame,mask=mask)#apply mask to already gray image (gray image to match the shape of mask) to only detect specific movement
        #for object dxetection
        c = max(contourfn(masked_frame,False), key = cv2.contourArea)#get the countour with max area
        (x,y,w,h)=cv2.boundingRect(c)#coordinates of a rect around it
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)#draw a rect based on coordinates
        a=int(((x+w)+x)/2)#mid distance of width
        b=int(((y+h)+y)/2)#mid distance of height
        cv2.line(frame1,(a,y),(a,y+h),(255,0,0),2) #vertical line
        cv2.line(frame1,(x,b),(x+w,b),(255,0,0),2) #horizontal line
        cv2.circle(frame1,(a,b),10,(255,0,0),2) #draw circle in middle 
        li_contour.append(c)#append the max area countour to list

        #for color detection
        c = max(contourfn(result,True), key = cv2.contourArea)
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
        a=int(((x+w)+x)/2)
        b=int(((y+h)+y)/2)
        cv2.circle(result,(a,b),10,(255,0,0),2)
        li_contour.append(c)



        print(contourIntersect(frame1, li_contour[0], li_contour[1]))#add motor control here 


    except ValueError as e:
        #print(e)
        cv2.imshow('Camera2', result)#show the resultant image
        cv2.imshow('Camera', frame1)
        
    

    cv2.imshow('Camera2', result)#show the resultant image
    cv2.imshow('Camera', frame1)

    ############################################################################################################################
    #frame update block
    ############################################################################################################################

    frame1=frame2 #change of latest frame to previous
    isTrue, frame2 = cam.read() #get new frame 

    frame_1=frame_2
    isTrue, frame_2 = cam.read()

    ############################################################################################################################
    #misc and break fn
    ############################################################################################################################
    li_contour.clear()#clear the list
    #time.sleep(.05)#sleep timer for a better target retention


    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
