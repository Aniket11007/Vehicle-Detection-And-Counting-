import cv2
import numpy as np
from numpy.core.defchararray import center


#Web Camera

capt= cv2.VideoCapture('video.mp4')



min_width_rect = 80 

min_height_rect = 80 

count_line =550

#Initialize Substraction algorithm
step = cv2.bgsegm.createBackgroundSubtractorMOG()
#It will subtract the image from the background
#So it will  Subtract vehicle from the background in the video

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = [] #in this list we'll append the count
offset = 6 #Allowable error between pixel 
counter_vehicle=0
while True:
    #To read the video
    frame_return,frameA = capt.read()

    # convert to gray scale of each frames 


    grey = cv2.cvtColor(frameA,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey,(3,3),5)    #(3,3) is default

    #Applying substraction for rach frame

    vid_sub = step.apply(blur)
    dil = cv2.dilate(vid_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))  #it just give structure or  shape the algorithm
    dil_data = cv2.morphologyEx(dil,cv2.MORPH_CLOSE,kernel)  #This Function just give shape 
    dil_data = cv2.morphologyEx(dil_data,cv2.MORPH_CLOSE,kernel)  #This Function just give shape 

    counter_shape,h = cv2.findContours(dil_data,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



    cv2.line(frameA,(25,count_line),(1160,count_line),(255,127,0),8) #It will draw a line on a video frame

    for (i,c) in enumerate(counter_shape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (w>=min_height_rect)

        if not validate_counter:
            continue

        cv2.rectangle(frameA,(x,y),(x+w,y+h),(0,255,0),2)  
        
        cv2.putText(frameA,"VEHICLE:"+str(counter_vehicle),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),5)


        center = center_handle(x,y,w,h)
        detect.append(center)

        cv2.circle(frameA,center,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line+offset) and y>(count_line-offset):
                counter_vehicle+=1
            cv2.line(frameA,(25,count_line),(1160,count_line),(0,127,255),3)
            detect.remove((x,y)) 
            print("Total Vehicle Count:"+str(counter_vehicle))    


    #add text to the upper right corner total vechile count
    cv2.putText(frameA,"TOTAL VEHICLE COUNT:"+str(counter_vehicle),(450,70),cv2.FONT_HERSHEY_PLAIN,2,(255,244,0),5)




    #cv2.imshow('Detector',dil_data)

    #After reading We need to show the video in the loop

    cv2.imshow('Video Original',frameA)

    if cv2.waitKey(1) ==13:
        break

cv2.destroyAllWindows()
capt.release()
    




