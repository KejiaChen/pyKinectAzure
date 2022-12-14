import os
import cv2
import numpy as np
from fastdlo.core import Pipeline
import sys
sys.path.append(os.path.join(sys.path[0],'..'))
from aruco_detection import ArucoDetector
import pykinect_azure as pykinect
import copy

def fastdlo_segmentation(img, pipe):
	# run the FASTDLO, return a array of size (IMG_H, IMG_W, 3)
	img_out, dlo_mask_pointSet, dlo_path = pipe.run(source_img=img, mask_th=15)
		
	for i in range(len(dlo_path)):
		pos = dlo_path[i]
		pos = R.dot(pos) # coordinate btw image and path fitting
		if i != len(dlo_path)-1:
			pos2 = dlo_path[i+1]
			pos2 = R.dot(pos2)
			# detect whether the points in between are too far away, 
			# if not, then connect them with line segment
			if np.sqrt(((pos[0]-pos2[0])**2)+((pos[1]-pos2[1])**2)) < Distance_threshold:
				cv2.line(img_out, pt1=pos, pt2=pos2, color=(255,0,0))
			else:
				pass
		else:
			pos2 = None
		# draw the path line marker
		cv2.drawMarker(img, tuple(pos), color=(255,0,0), markerType=3, markerSize=7, thickness=1)

	canvas = img.copy()
	# show the detected results with origin stream video together with weight 0.5 for each 
	canvas = cv2.addWeighted(canvas, 1.0, img_out, 0.5, 0.0)
	return canvas

if __name__ == "__main__":
	
	#########################
	# Set up the parameters #
	#########################
	'''
	# YOU MIGHT NEED TO TOUCH THIS PARAMETERS!
	'''
	IMG_COLOR_SIZE = [1080, 1920, 3]
	IMG_ROI_SIZE = [451, 440, 3]
	MASK_RED = {"lower": np.array([0, 70, 70]),
            	"upper": np.array([10, 255, 255])}

	COLOR_RANGE = [
		# ((0, 70, 70), (10, 255, 255)),
		((0, 70, 50), (10, 255, 255)),
		((170, 70, 50), (180, 255, 255)),
        # ((0, 43, 46),(10, 255, 255)), # red color range 1 HSV
        # ((156, 43, 46),(180, 255, 255)) # red color range 2 HSV
        # you can add as many colors as you would like
        # final detected merged color == color 1 + color 2 + ..
    ]
	R = np.array([[0,1],[1,0]]) # Rotation matrix for coordinate fitting
	Distance_threshold = 60 # distance detection threshold
	
	# weighting file name - NO NEED TO TOUCH
	ckpt_siam_name = "CP_similarity.pth"
	ckpt_seg_name = "CP_segmentation.pth"
	# get current run file path - NO NEED TO TOUCH
	script_path = os.getcwd()
	# get network weights <- need to create folder named 'weights' under main program folder, paste *.ckpt/*.pth files in it
	checkpoint_siam = os.path.join(script_path, "FASTDLO_IntelRealSense/weights/" + ckpt_siam_name)
	checkpoint_seg = os.path.join(script_path, "FASTDLO_IntelRealSense/weights/" + ckpt_seg_name)
	# load FASTDLO algorithm pipeline - NO NEED TO TOUCH
	p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_COLOR_SIZE[1], img_h=IMG_COLOR_SIZE[0], colorRange=COLOR_RANGE)
	# p_roi = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_ROI_SIZE[1], img_h=IMG_ROI_SIZE[0])
	
	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()
	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	# print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)
 
 	# Aruco Detector
	roi_detector = ArucoDetector("DICT_5X5_50")

	while True:

		# Get capture
		capture = device.update()

		# Get the color image from the capture
		ret, color_image = capture.get_color_image()
		_, colored_depth_image = capture.get_colored_depth_image()
		
		if not ret:
			continue

		# color_image = np.asanyarray(color_image)
		# roi_image = copy.deepcopy(color_image)
		# roi_vertices = roi_detector.roi_run(color_image, colored_depth_image)
		# roi_color_image = roi_image[roi_detector.vertices[0][1]:roi_detector.vertices[2][1], roi_detector.vertices[0][0]:roi_detector.vertices[2][0]]
		# roi_depth_image = colored_depth_image[roi_detector.vertices[0][1]:roi_detector.vertices[2][1], roi_detector.vertices[0][0]:roi_detector.vertices[2][0]]
		# cv2.imshow("Image", roi_color_image)
		# key = cv2.waitKey(1)

		dlo_segment = fastdlo_segmentation(color_image, p)
		cv2.namedWindow("pyKinect_colorimage", cv2.WINDOW_AUTOSIZE) # set up the window to display the results
		cv2.imshow("pyKinect_colorimage", dlo_segment)	

		# dlo_segment_roi = fastdlo_segmentation(color_image, p)
		# cv2.namedWindow("pyKinect_roiimage", cv2.WINDOW_AUTOSIZE) # set up the window to display the results
		# cv2.imshow("pyKinect_roiimage", dlo_segment_roi)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'): 
			break
