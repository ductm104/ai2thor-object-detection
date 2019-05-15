import numpy as np
import cv2

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

class Yolo:

	def __init__(self):

		model_path = './models/yolov3.weights'
		config_path = './models/yolov3.cfg'

		self.model = cv2.dnn.readNetFromDarknet(config_path, model_path)
		self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
		print('Yolov3 model is loaded, using CPU.')

		self.threshold = 0.5
		# self.nmsThreshold = 0.4  # Non-maximum suppression threshold
		self.INPUT_WIDTH = 416
		self.INPUT_HEIGHT = 416

		# determine only the *output* layers that we need from YOLO
		ln = self.model.getLayerNames()
		self.outputLayers = [ln[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

	def detect(self, frame):
		blob = cv2.dnn.blobFromImage(frame, 
				1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT),
				swapRB=True, crop=False)

		self.model.setInput(blob)
		outs = self.model.forward(self.outputLayers)

		return self.postprocess(frame, outs)
	
	def postprocess(self, frame, outs):
		frameHeight, frameWidth, _ = frame.shape

		classIds = []
		confidences = []
		boxes = []
		# Scan through all the bounding boxes output from the network and keep only the
		# ones with high confidence scores. Assign the box's class label as the class 
		# with the highest score.
		for out in outs:
			for detection in out:
				scores = detection[5:]
				classId = np.argmax(scores)
				confidence = scores[classId]

				if confidence >= self.threshold:
					center_x = int(detection[0] * frameWidth)
					center_y = int(detection[1] * frameHeight)

					width = int(detection[2] * frameWidth)
					height = int(detection[3] * frameHeight)

					left = max(0, int(center_x - width / 2))
					top = max(0, int(center_y - height / 2))
					# x2 = min(int(center_x + width / 2), frameWidth)
					# y2 = min(int(center_y + height / 2), frameHeight)

					classIds.append(classId)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])

		return boxes, classIds, confidences

if __name__ == '__main__':
	img = cv2.imread('test_image.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	labels_path = './models/coco.names'
	labels_to_names = open(labels_path).read().strip().split('\n')
	yolo = Yolo()

	boxes, labels, scores = yolo.detect(img)

	for box, label, score in zip(boxes, labels, scores):
		print(box, label, score)
		color = label_color(label)
		draw_box(img, box, color=color)
		caption = "{} {:.3f}".format(labels_to_names[label], score)
		draw_caption(img, box, caption)

	cv2.imshow('out', img)
	cv2.waitKey(0)