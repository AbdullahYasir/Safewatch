"""
- 1NEX
    - https://www.1nex.co/
    - Mir Abdullah Yasir <mirabdullahyaser@gmail.com>
    - Copyright: Copyright (C) 1NEX
- predict_video.py version 1.0.0
"""

# import the necessary packages
from spacy import load
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import logging
import time
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
		python .\predict_video.py --model .\model\activity.model --label-bin .\output\lb.pickle --input .\example_clips\football_01.mp4 --output .\output\football_01.avi --size 1 --loglevel ERROR
	"""
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", required=True,
		help="path to trained serialized model")
	ap.add_argument("-l", "--label-bin", required=True,
		help="path to  label binarizer")
	ap.add_argument("-i", "--input", required=True,
		help="path to our input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to our output video")
	ap.add_argument("-s", "--size", type=int, default=128,
		help="size of queue for averaging")
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


def load_model():
	# load the trained model and label binarizer from disk
	logger.info("loading model and label binarizer...")
	model = load_model(args["model"])
	lb = pickle.loads(open(args["label_bin"], "rb").read())
	# initialize the image mean for mean subtraction along with the
	# predictions queue
	mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
	Q = deque(maxlen=args["size"])
	return model, lb, mean, Q


def read_and_predict(model, lb, mean, Q):
	# initialize the video stream, pointer to output video file, and
	# frame dimensions
	vs = cv2.VideoCapture(args["input"])
	writer = None
	(W, H) = (None, None)
	# loop over frames from the video file stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]


		# clone the output frame, then convert it from BGR to RGB
		# ordering, resize the frame to a fixed 224x224, and then
		# perform mean subtraction
		output = frame.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (224, 224)).astype("float32")
		frame -= mean


		# make predictions on the frame and then update the predictions
		# queue
		preds = model.predict(np.expand_dims(frame, axis=0))[0]
		Q.append(preds)
		# perform prediction averaging over the current history of
		# previous predictions
		results = np.array(Q).mean(axis=0)
		i = np.argmax(results)
		label = lb.classes_[i]


		# draw the activity on the output frame
		text = "activity: {}".format(label)
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
			1.25, (0, 255, 0), 5)
		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)
		# write the output frame to disk
		writer.write(output)
		# show the output image
		cv2.imshow("Output", output)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	# release the file pointers
	logger.info("cleaning up...")
	writer.release()
	vs.release()


def main():
	model, lb, mean, Q = load_model()
	read_and_predict(model, lb, mean, Q)


if __name__ == "__main__":
	script_begin_time = time.time()
	args = parse_args()
	logger = init_logger(args)

	main()

	script_end_time = time.time()
	script_execution_time = script_end_time - script_begin_time
	logger.info('Process finished successfully in {}'.format(hms(script_execution_time)))