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
from threadClass import ThreadWithReturnValue as thread

files = glob.glob('output/*.png')
for f in files:
	os.remove(f)

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

# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
# load our YOLO object detector trained on COCO dataset (80 classes)
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
	counter = 0
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
	bboxes = []

	if len(outBoxes) > 0:
		for box in outBoxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))
			bbox = [x, y, w, h, box[4]]
			bboxes.append(bbox)
			cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

	if len(allObjects) == 0 and len(outBoxes) > 0:
		for bbox in bboxes:
			#print(bbox[4])
			box = ObjectTracking(bbox[4])
			bbox = bbox[:-1]
			box.createNewID(bbox, allObjects)
			allObjects.append(box)
	else:
		for object1 in allObjects:
			ex = False
			for bbox in bboxes:
				b = bbox
				bbox = bbox[:-1]
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
							cc += 1

			#draw a bounding box rectangle and label on the image
			#object1.getcount() == 30 and
			if object1.getClassName() == "person":
				counter += 1
			bbox = object1.getbbox()
			cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
			text = 'id = {} classID = {} counter = {}'.format(object1.id, object1.getClassName(), object1.getcount())
			cv2.putText(frame, text, (object1.bbox[0], object1.bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

	if len(bboxes) > 0:
		for bbox in bboxes:
			#print(bbox[4])
			box = ObjectTracking(bbox[4])
			bbox = bbox[:-1]
			box.createNewID(bbox, allObjects)
			allObjects.append(box)


	print("--- %s seconds ---" % (time.time() - start_time))

	if resolution == '4k':
		text = str(len(allObjects)) + ' ' + str(counter)
		font = cv2.FONT_HERSHEY_SIMPLEX
		#cv2.putText(frame, str(len(allObjects)), (50,50), font, 2, (243, 132, 68), 4, cv2.LINE_AA)
		cv2.putText(frame, text, (100,200), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

	cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame

	# write the output frame to disk
	writer.write(frame)

	# increase frame index
	frameIndex += 1

# release the file pointers
print("[INFO] cleaning up...")
files = glob.glob('output/*.png')
for f in files:
	os.remove(f)
yolo.close_session()
writer.release()
vs.release()
