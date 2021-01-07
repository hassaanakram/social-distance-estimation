import cv2
import numpy as np

# Create empty list of points for coords
list_points = list()

# Define the callback function that we are going to use to get coords
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param = cv2.circle(param,(x,y),5,(0,255,0),-1)
        print("Left Mouse click at (", x, ",", y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        param = cv2.circle(param,(x,y),5,(0,255,0),-1)
        print("Right Mouse click at (", x, ",", y, ")")
        list_points.append([x,y])

        
def calibrate(frame):
    # Create a blank window
    global list_points
    list_points = list()
    print("Calibration started")
    windowName = 'MouseCallback'
    cv2.namedWindow(windowName)

    # Get the size of input frame
    height, width, _ = frame.shape
    print("Width,Height of frame: ({},{})".format(width, height))
    # Bind callback function to window
    cv2.setMouseCallback(windowName, CallBackFunc, frame)

    # Check if 4 points have been saved
    while (True):
        cv2.imshow(windowName, frame)
        cv2.waitKey(10) & 0xFF
        
        if len(list_points) == 4:
            break
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    
    # Calculating projection matrix
    src_pts = np.float32(list_points)
    dest_pts = np.float32([[0,0],[width,0],[0,height],[width,height]])

    matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
    print("Image to birds eye matrix: {}".format(matrix))
    inv_matrix = cv2.getPerspectiveTransform(dest_pts, src_pts)
    print("Birds eye to image matrix: {}".format(inv_matrix))
    return matrix, inv_matrix, list_points
