import cv2
import numpy as np
import time
import argparse
from pathlib import Path 

parser = argparse.ArgumentParser()
parser.add_argument('img_folder', default='data/all_images/image_moderation_images/')
parser.add_argument('--min_conf', type=int, default = 0.95)

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

def generate_mask(image,net, min_conf=0.95,show=False):
	orig = image.copy()
	(H, W) = image.shape[:2]
	
	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)
	
	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()
	
	# show timing information on text prediction
	print("[INFO] text detection took {:.6f} seconds".format(end - start))

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
			# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < min_conf:
				continue
	
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
	
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes

	boxes = non_max_suppression(np.array(rects), probs=confidences)
	masked = np.zeros(orig.shape) 
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		
		masked[startY:endY,startX:endX,:] = 255
		# draw the bounding box on the image
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	# show the output image
	if  show:
		cv2.imshow("Text Detection", orig)
		cv2.imshow("Masked", masked)
		cv2.waitKey()
	
	return masked

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
		# if there are no boxes, return an empty list
		if len(boxes) == 0:
			return []

		# if the bounding boxes are integers, convert them to floats -- this
		# is important since we'll be doing a bunch of divisions
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")

		# initialize the list of picked indexes
		pick = []

		# grab the coordinates of the bounding boxes
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]

		# compute the area of the bounding boxes and grab the indexes to sort
		# (in the case that no probabilities are provided, simply sort on the
		# bottom-left y-coordinate)
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = y2

		# if probabilities are provided, sort on them instead
		if probs is not None:
			idxs = probs

		# sort the indexes
		idxs = np.argsort(idxs)

		# keep looping while some indexes still remain in the indexes list
		while len(idxs) > 0:
			# grab the last index in the indexes list and add the index value
			# to the list of picked indexes
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)

			# find the largest (x, y) coordinates for the start of the bounding
			# box and the smallest (x, y) coordinates for the end of the bounding
			# box
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])

			# compute the width and height of the bounding box
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)

			# compute the ratio of overlap
			overlap = (w * h) / area[idxs[:last]]

			# delete all indexes from the index list that have overlap greater
			# than the provided overlap threshold
			idxs = np.delete(idxs, np.concatenate(([last],
				np.where(overlap > overlapThresh)[0])))

		# return only the bounding boxes that were picked
		return boxes[pick].astype("int")

if __name__ == "__main__":
	args = parser.parse_args()
	show = True
	min_conf = args.min_conf
	img_path = Path(args.img_folder)/'1864109.jpg'

	print("[INFO] loading EAST text detector...")	
	net = cv2.dnn.readNet('frozen_east_text_detection.pb')
	image = cv2.imread(img_path)
	generate_mask(image, min_conf, show)
