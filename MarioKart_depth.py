import cv2
import numpy as np
#from pyvesc import VESC
import time
import depthai as dai
import blobconverter

# BOOST = "boost"
# BOOST_RPM = 8000

# SLOW = "slow"
# SLOW_RPM = 3000

# STOP = "stop"
# STOP_RPM = 0

# NORMAL = "normal"
# NORMAL_RPM = 5000


# SERIAL_PORT = "/dev/ttyACM0"

# def set_rpm(rpm):
#     with VESC(serial_port=SERIAL_PORT) as motor:
#         motor.set_duty_cycle(.02)
#         motor.set_rpm(rpm)
#     return
        

# def set_state(state):
#     if state == BOOST:
#         set_rpm(BOOST_RPM)
#         time.sleep(2)
#     elif state == SLOW:
#         set_rpm(SLOW_RPM)
#         time.sleep(2)
#     elif state == STOP:
#         set_rpm(STOP_RPM)
#         time.sleep(2)

#     set_rpm(NORMAL_RPM)

#     return
    


def contour_detection(imageFrame,depth_data): 
    # set_state(NORMAL)

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Calculate total pixels in frame
    total_pixels = np.prod(imageFrame.shape[:2])
  
    # Set range for orange color and 
    # define mask
    orange_lower = np.array([5, 50, 50], np.uint8)
    orange_upper = np.array([15, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Calculate porportion to frame
    orange_pixels = cv2.countNonZero(orange_mask)
    orange_percentage = orange_pixels / total_pixels

    if orange_percentage >= 0.4:
        print("ORANGE")
        # set_state(STOP)
  
    # Set range for pink color and 
    # define mask
    pink_lower = np.array([140, 50, 50], np.uint8)
    pink_upper = np.array([180, 255, 255], np.uint8)
    pink_mask = cv2.inRange(hsvFrame, pink_lower, pink_upper)

    # Calculate porportion to frame
    pink_pixels = cv2.countNonZero(pink_mask)
    pink_percentage = pink_pixels / total_pixels

    if pink_percentage >= 0.4:
        print("PINK")
        # set_state(SLOW)
  
    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 50, 50], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Calculate porportion to frame
    blue_pixels = cv2.countNonZero(blue_mask)
    blue_percentage = blue_pixels / total_pixels

    if blue_percentage >= 0.4:
        print("BLUE")
        # set_state(BOOST)
      
    kernel = np.ones((5, 5), "uint8")
      
    orange_mask = cv2.dilate(orange_mask, kernel)
    pink_mask = cv2.dilate(pink_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

   
    # Creating contour to track orange color
    contours, hierarchy = cv2.findContours(orange_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 40000):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (0, 0, 255), 2)
              
            cv2.putText(imageFrame, "Orange Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))    
  
    # Creating contour to track pink color
    contours, hierarchy = cv2.findContours(pink_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 40000):
            x, y, w, h = cv2.boundingRect(contour)
            depth_box = np.array(depth_data)
            depth_box = depth_box[x:x+w,y:y+w]
            depth_result = np.mean(depth_box)

            # rect_bottom_left = (x, y)  # (x, y)
            # rect_top_right = (x+w, y+h)
            # depth_values = []
            # for y in range(rect_bottom_left[1], rect_top_right[1]):
            #     for x in range(rect_bottom_left[0], rect_top_right[0]):
            #         depth = imageFrame[y, x]
            #         if depth.any():
            #              depth_values.append(depth)

            # if len(depth_values) > 0:
            #     avg_depth = np.mean(depth_values)
            # else:
            #     avg_depth = 0

            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h),
                                       (0, 255, 0), 2)
            cv2.putText(imageFrame, "Pink Colour " + str("avg_depth"), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0))
     
    # Creating contour to track blue color
    contours, _ = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 40000):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)
              
            cv2.putText(imageFrame, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))

    return imageFrame




# Initialize the webcam
#cap = cv2.VideoCapture(0)
#Using DepthAI
pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName('rgb')
cam_rgb.preview.link(xout_rgb.input)

device = dai.Device(pipeline)
output = device.getOutputQueue(name='rgb', maxSize=4, blocking=False)

# left and right cam
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(245)
stereo.setInputResolution(640, 480)
cam_left.out.link(stereo.left)
cam_right.out.link(stereo.right)
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName('depth')
stereo.depth.link(xout_depth.input)
device = dai.Device(pipeline)
output_depth = device.getOutputQueue(name='depth', maxSize=1, blocking=True)


while True:
    # Capture a frame from the webcam
    depth_frame = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    disparity = depth_frame.get()
    #ret, frame = cap.read()
    in_rgb = output.get()
    frame_rgb = in_rgb.getCvFrame()
    #depths_data = disparity.getCvFrame()
    # Perform edge detection on the frame
    result = contour_detection(frame_rgb,disparity)
    
    # Show the result
    # cv2.imshow('Result', result)
    #cv2.imshow('RGB', result)
    cv2.imshow('RGB', result)
    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
#cap.release()
cv2.destroyAllWindows()