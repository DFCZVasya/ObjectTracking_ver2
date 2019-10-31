# import the necessary packages
import sys
from PIL import Image
from yolo import YOLO
import numpy as np
import argparse
import imutils
import tensorflow as tf
from keras import backend as K
from keras.utils import multi_gpu_model
import time
import cv2
import os
import glob
from objectTracking import ObjectTracking

files = glob.glob('output/*.png')
for f in files:
	os.remove(f)

#from sort import *
#tracker = Sort()
#memory = {}

#resolution = input('Enter Resolution: ')
resolution = '4k'
if resolution == '4k':
	line = [(2000, 0), (2050,3840)]
elif resolution == 'hd':
	line = [(350, 0), (350,720)] # 4k 1750 0 1750 2160
else:
	line = [(700, 0),(700,1080)]
counter = 0
count =0
counterb = 0
counterh = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
#ap.add_argument("-y", "--yolo", required=True,
#	help="base path to YOLO directory")
#ap.add_argument("-c", "--confidence", type=float, default=0.5,
#	help="minimum probability to filter weak detections")
#ap.add_argument("-t", "--threshold", type=float, default=0.3,
#	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Return true if line segments AB and CD intersect
#def intersect(A,B,C,D):
#	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#def ccw(A,B,C):
#	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
#LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
#np.random.seed(42)
#COLORS = np.random.randint(0, 255, size=(200, 3),
#	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
#print("[INFO] loading YOLO from disk...")
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#ln = net.getLayerNames()
#ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

yolo = YOLO()

frameIndex = 0
#howto = str(input("Please input what need found: "))
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
allObjects = []
# loop over frames from the video file stream
while True:
	start_time = time.time()
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	###########################################################################################
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	image = Image.fromarray(frame)
	#image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

	outBoxes = yolo.detect_image(image)
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	#blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
	#	swapRB=True, crop=False)
	#net.setInput(blob)
	###################################################################
	#layerOutputs = net.forward(ln)
	####################################################################
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	#boxes = []
	#confidences = []
	#classIDs = []

	# loop over each of the layer outputs
	#for output in layerOutputs:
		# loop over each of the detections
		#for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			#scores = detection[5:]
			#classID = np.argmax(scores)
			#confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			#if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				#box = detection[0:4] * np.array([W, H, W, H])
				#(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				#x = int(centerX - (width / 2))
				#y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				#boxes.append([x, y, int(width), int(height)])
				#confidences.append(float(confidence))
				#classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	#idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	#dets = []
	#if len(idxs) > 0:
		# loop over the indexes we are keeping
		#for i in idxs.flatten():
		#	(x, y) = (boxes[i][0], boxes[i][1])
		#	(w, h) = (boxes[i][2], boxes[i][3])
		#	dets.append([x, y, x+w, y+h, confidences[i]])

	#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	#dets = np.asarray(dets)
	#tracks = tracker.update(dets)

	#boxes = []
	#indexIDs = []
	#c = []
	bboxes = []

	#for track in tracks:
	#	boxes.append([track[0], track[1], track[2], track[3]])
	#	indexIDs.append(int(track[4]))

	if len(outBoxes) > 0:
		#i = int(0)
		for box in outBoxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))
			bbox = [x, y, w, h, box[4]]
			bboxes.append(bbox)
			#i += 1
			#cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

	if len(allObjects) == 0 and len(outBoxes) > 0:
		for bbox in bboxes:
			box = ObjectTracking(bbox[4])
			bbox = bbox[:-2]
			box.createNewID(bbox, allObjects)
			allObjects.append(box)
	else:
		for object1 in allObjects:
			ex = False
			for bbox in bboxes:
				b = bbox
				bbox = bbox[:-2]
				k = object1.getIntersection(bbox)
				if k > 0.5:
					ex = False
					object1.tracking(bbox, k, ex)
					bboxes.remove(b)
					break
				else:
					ex = True

			if ex == True:
				idforDelete = object1.tracking(0, k, ex)
				if idforDelete != 0:
					cc = 0
					for object2 in allObjects:
						if object2.id == idforDelete:
							allObjects.pop(cc)
							#print('cc = ' + str(cc))
							cc += 1

			#draw a bounding box rectangle and label on the image

			#color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (object1.bbox[0], object1.bbox[1]), (object1.bbox[2], object1.bbox[3]), (255,0,0), 2)

			#color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			#cv2.rectangle(frame, (object1.bbox[0], object1.bbox[1]), (object1.bbox[2] - object1.bbox[0], object1.bbox[3] - object1.bbox[1]), (255,0,0), 2)

			#text = "{}: classID: {}  confidence: {:.4f}".format(object1.classID, object1.id, object1.probability)
			#text = "{}".format(indexIDs[i])
			#cv2.putText(frame, text, (object1.bbox[0], object1.bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

	if len(bboxes) > 0:
		for bbox in bboxes:
			box = ObjectTracking(bbox[4])
			bbox = bbox[:-2]
			box.createNewID(bbox, allObjects)
			allObjects.append(box)
	print("--- %s seconds ---" % (time.time() - start_time))




	if resolution == '4k':

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, str(len(allObjects)), (50,50), font, 2, (243, 132, 68), 4, cv2.LINE_AA)

	cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		#if total > 0:
			#elap = (end - start)
			#print("[INFO] single frame took {:.4f} seconds".format(elap))
			#print("[INFO] estimated total time to finish: {:.4f}".format(
			#	elap * total))

	# write the output frame to disk
	writer.write(frame)

	# increase frame index
	frameIndex += 1

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
