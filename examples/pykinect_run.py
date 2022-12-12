import os
import cv2
import numpy as np
from FASTDLO_IntelRealSense.fastdlo.core import Pipeline

import pykinect_azure as pykinect

if __name__ == "__main__":
    
    #########################
    # Set up the parameters #
    #########################
    '''
    # YOU MIGHT NEED TO TOUCH THIS PARAMETERS!
    '''
    IMG_W = 640 # image size
    IMG_H = 480
    R = np.array([[0,1],[1,0]]) # Rotation matrix for coordinate fitting
    Distance_threshold = 60 # distance detection threshold
    
    # weighting file name - NO NEED TO TOUCH
    ckpt_siam_name = "CP_similarity.pth"
    ckpt_seg_name = "CP_segmentation.pth"
    # get current run file path - NO NEED TO TOUCH
    script_path = os.getcwd()
    # get network weights <- need to create folder named 'weights' under main program folder, paste *.ckpt/*.pth files in it
    checkpoint_siam = os.path.join(script_path, "weights/" + ckpt_siam_name)
    checkpoint_seg = os.path.join(script_path, "weights/" + ckpt_seg_name)
    # load FASTDLO algorithm pipeline - NO NEED TO TOUCH
    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)

	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()
	# Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	# print(device_config)

	# Start device
    device = pykinect.start_device(config=device_config)

    while True:

		# Get capture
        capture = device.update()

		# Get the color image from the capture
        ret, color_image = capture.get_color_image()
        
        if not ret:
            continue

        color_image = np.asanyarray(color_image.get_data())
        # run the FASTDLO, return a array of size (IMG_H, IMG_W, 3)
        img_out, dlo_mask_pointSet, dlo_path = p.run(source_img=color_image, mask_th=77)
        
        for i in range(len(dlo_path)):
            pos = dlo_path[i]
            pos = R.dot(pos) # coordinate btw image and path fitting
            if i != len(dlo_path)-1:
                pos2 = dlo_path[i+1]
                pos2 = R.dot(pos2)
                # detect whether the points in between are too far away, 
                # if not, then connect them with line segment
                if np.sqrt(((pos[0]-pos2[0])**2)+((pos[1]-pos2[1])**2)) < Distance_threshold:
                    cv2.line(color_image, pt1=pos, pt2=pos2, color=(255,0,0))
                else:
                    pass
            else:
                pos2 = None
            # draw the path line marker
            cv2.drawMarker(color_image, tuple(pos), color=(255,0,0), markerType=3, markerSize=7, thickness=1)

        canvas = color_image.copy()
        # show the detected results with origin stream video together with weight 0.5 for each 
        canvas = cv2.addWeighted(canvas, 1.0, img_out, 0.5, 0.0) 

        cv2.namedWindow("pyKinect_colorimage", cv2.WINDOW_AUTOSIZE) # set up the window to display the results
        cv2.imshow("pyKinect_colorimage", canvas)	

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
        	break
