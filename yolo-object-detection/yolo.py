"""
- 1NEX
    - https://www.1nex.co/
    - Mir Abdullah Yasir <mirabdullahyaser@gmail.com>
    - Copyright: Copyright (C) 1NEX
- yolo.py version 1.0.0
"""

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import logging
import sys

logger = None


def hms(seconds):
    """
    Converts time in seconds to Human Readable format
    :param seconds: time elapsed to convert
    :type seconds: int,float
    :return: formatted time string
    :rtype: str

    """
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = seconds % 3600 % 60
    return '{:02d}h:{:02d}m:{:.2f}s'.format(h, m, s)


def parse_args():
	use = """
	Version: 1.0
	Copyright: 1NEX
	Example:
		Sample Commands:
		python .\yolo.py -i .\images\00000544.jpg -y .\yolo-coco\ --loglevel ERROR
	"""
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")
	ap.add_argument('-log', '--loglevel', dest='loglevel', action='store',
		default='DEBUG', help='Loglevel of log file')
	args = vars(ap.parse_args())
	return args


def init_logger(arguments):
    numeric_level = getattr(logging, arguments['loglevel'].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % arguments['loglevel'])
    # noinspection PyArgumentList
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)-8s %(message)s', level=numeric_level,
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger('SafeWatch')


def load_class_labels():
	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")
	return COLORS, LABELS


def load_model_weights():
	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	# load our YOLO object detector trained on COCO dataset (80 classes)
	logger.info("loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	return net

def pass_image_to_model(net):
	# load our input image and grab its spatial dimensions
	image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# show timing information on YOLO
	logger.info("YOLO took {:.6f} seconds".format(end - start))
	return layerOutputs, image, H, W

def create_bounding_boxes(layerOutputs, H, W):
	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	return boxes, confidences, classIDs


def apply_non_maxima_suppression(boxes, confidences):
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	return idxs


def show_bounding_boxes(idxs, boxes, confidences, classIDs, image, COLORS, LABELS):
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)


def main():
	COLORS, LABELS = load_class_labels()
	net = load_model_weights()
	layerOutputs, image, H, W = pass_image_to_model(net)
	boxes, confidences, classIDs = create_bounding_boxes(layerOutputs, H, W)
	idxs = apply_non_maxima_suppression(boxes, confidences)
	show_bounding_boxes(idxs, boxes, confidences, classIDs, image, COLORS, LABELS)


if __name__ == "__main__":
	script_begin_time = time.time()
	args = parse_args()
	logger = init_logger(args)

	main()

	script_end_time = time.time()
	script_execution_time = script_end_time - script_begin_time
	logger.info('Process finished successfully in {}'.format(hms(script_execution_time)))