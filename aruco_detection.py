'''
reference: https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/ by Adrian Rosebrock 
'''

import numpy as np
import cv2
import argparse
import sys
import pykinect_azure as pykinect
import threading
import copy

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

class ArucoDetector:
	def __init__(self, tag_type:str) -> None:
		self.vertices = {}
		self.marker_size = None
		self.tag_type = tag_type
		if ARUCO_DICT.get(self.tag_type, None) is None:
			print("[INFO] ArUCo tag of '{}' is not supported".format(tag_type))
			self.tag_type = "DICT_5X5_50"
		# load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
		print("[INFO] detecting '{}' tags...".format(self.tag_type))
		
		self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[self.tag_type]) # 5x5 for environmental markers to define ROI
		self.aruco_params = cv2.aruco.DetectorParameters_create()
	
	def vertice_distance(self, v1, v2):
		vec = np.array(list(v1)) - np.array(list(v2))
		dist = np.linalg.norm(vec)
		return dist
		
	def aruco_detect(self, image):
		# (corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict, self.arucoParams)
		(corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)
			
		# verify *at least* one ArUco marker was detected
		if len(corners) > 0:
			# flatten the ArUco IDs list
			ids = ids.flatten()
			# loop over the detected ArUCo corners
			for (markerCorner, markerID) in zip(corners, ids):
				# extract the marker corners (which are always returned in
				# top-left, top-right, bottom-right, and bottom-left order)
				corners = markerCorner.reshape((4, 2))
				(topLeft, topRight, bottomRight, bottomLeft) = corners
				# convert each of the (x, y)-coordinate pairs to integers
				topRight = (int(topRight[0]), int(topRight[1]))
				bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
				bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
				topLeft = (int(topLeft[0]), int(topLeft[1]))
				# compute the center (x, y)-coordinates of the ArUco marker
				cX = int((topLeft[0] + bottomRight[0]) / 2.0)
				cY = int((topLeft[1] + bottomRight[1]) / 2.0)
				self.vertices[markerID] = (cX, cY)

				# draw the bounding box of the ArUCo detection
				cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
				dist_1 = self.vertice_distance(topLeft, topRight)
				cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
				dist_2 = self.vertice_distance(topRight, bottomRight)
				cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
				dist_3 = self.vertice_distance(bottomRight, bottomLeft)
				cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
				dist_4 = self.vertice_distance(bottomLeft, topLeft)
				# draw the center (x, y)-coordinates of the ArUco marker
				cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
				# draw the ArUco marker ID on the image
				cv2.putText(image, str(markerID),
					(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (0, 255, 0), 2)
				# pixel size of the marker
				self.marker_size = (dist_1 + dist_2 + dist_3 + dist_4)/4
				# print("[INFO] ArUco marker ID: {}".format(markerID))
				# # show the output image
				# cv2.imshow("Image", image)
				# # cv2.waitKey(0)
				# # Press q key to stop
				# if cv2.waitKey(1) == ord('q'): 
				#     break
				
			# cv2.line(image, self.vertices[0], self.vertices[1], (255, 0, 0), 2)
			# cv2.line(image, self.vertices[1], self.vertices[2], (255, 0, 0), 2)
			# cv2.line(image, self.vertices[2], self.vertices[3], (255, 0, 0), 2)
			# cv2.line(image, self.vertices[3], self.vertices[0], (255, 0, 0), 2)
			
			# # print(VERTICES)
			
			# # show the output image
			# cv2.imshow("Image", image)
			
			# # cv2.waitKey(0)
			# # Press q key to stop
			# key = cv2.waitKey(1)
			# if key == ord('c'):
			#     cv2.destroyWindow("Image")
 
	def direction_run(self, start_id, target_id, color_image):
		direction_image = copy.deepcopy(color_image)
		self.aruco_detect(direction_image)
		if start_id in self.vertices and target_id in self.vertices:
			pixel_direction = np.array(list(self.vertices[target_id])) - np.array(list(self.vertices[start_id]))
			return pixel_direction/np.linalg.norm(pixel_direction)
			
	def clip_run(self, color_image):
		clip_image = copy.deepcopy(color_image)
		self.aruco_detect(clip_image)
  		# show the output image
		cv2.imshow("Fixture Image", clip_image)

	def roi_run(self, color_image, colored_depth_image):
		roi_image = copy.deepcopy(color_image)
		self.aruco_detect(roi_image)
				
		cv2.line(roi_image, self.vertices[0], self.vertices[1], (255, 0, 0), 2)
		cv2.line(roi_image, self.vertices[1], self.vertices[2], (255, 0, 0), 2)
		cv2.line(roi_image, self.vertices[2], self.vertices[3], (255, 0, 0), 2)
		cv2.line(roi_image, self.vertices[3], self.vertices[0], (255, 0, 0), 2)
				
		# print(VERTICES)
				
		# show the output image
		cv2.imshow("ROI Image", roi_image)
				
		# cv2.waitKey(0)
		# Press q key to stop
		# key = cv2.waitKey(1)
		# if key == ord('c'):
		#     cv2.destroyWindow("Image")
				
		# # Define Region of Interest
		# roi_color_image = roi_image[self.vertices[0][1]:self.vertices[2][1], self.vertices[0][0]:self.vertices[2][0]]
		# roi_depth_image = colored_depth_image[self.vertices[0][1]:self.vertices[2][1], self.vertices[0][0]:self.vertices[2][0]]
				
		# cv2.imshow('cropped Color ROI Image', roi_color_image)
		# cv2.imshow('cropped Depth ROI Image', roi_depth_image)
				
		# calibration.convert_2d_to_3d(source_point2d=self.vertices[0],
		#                              source_depth=depth_image[self.vertices[0]],
		#                              )
				
		# # Press q key to stop
		# key = cv2.waitKey(1)
		# if key == ord('q'):
		#     print("exit")
		#     break
	
 
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--type", type=str, default="DICT_5X5_50", help="type of ArUCo tag to detect")
	args = vars(ap.parse_args())
	
	roi_detector = ArucoDetector("DICT_5X5_50")
	clip_detector = ArucoDetector("DICT_4X4_50")
	ee_detector = ArucoDetector("DICT_4X4_50")
	
	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	# print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)
	# cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
	
	calibration = device.calibration
	
	while True:
		capture = device.update()
		ret, color_image = capture.get_color_image()
		_, transformed_depth_image = capture.get_transformed_depth_image()
		_, transformed_colored_depth_image = capture.get_transformed_colored_depth_image()
		_, colored_depth_image = capture.get_colored_depth_image()
		
		if not ret:
			continue
		else: 
			roi_image = copy.deepcopy(color_image)
   
			# pixel_direction = roi_detector.direction_run(start_id=9, target_id=5, color_image=roi_image)
			# robot_direction = np.array([round(pixel_direction[1]), round(pixel_direction[0])])
			# cv2.imshow("Image", roi_image)
			# key = cv2.waitKey(1)
   
			# # roi_thread = threading.Thread(target=roi_detector.roi_run, args=(color_image,))
			roi_vertices = roi_detector.roi_run(color_image, colored_depth_image)
			# # clip_detector.clip_run(color_image)

			roi_color_image = roi_image[roi_detector.vertices[0][1]:roi_detector.vertices[2][1], roi_detector.vertices[0][0]:roi_detector.vertices[2][0]]
			roi_depth_image = colored_depth_image[roi_detector.vertices[0][1]:roi_detector.vertices[2][1], roi_detector.vertices[0][0]:roi_detector.vertices[2][0]]
			print(np.shape(roi_color_image))
			cv2.imshow("Image", roi_color_image)
			key = cv2.waitKey(1)

