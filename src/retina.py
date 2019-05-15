import keras
import cv2
import numpy as np
import tensorflow as tf

from keras_retinanet.keras_retinanet import models
from keras_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.keras_retinanet.utils.colors import label_color

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(get_session())

class Retina:

	def __init__(self):
		model_path = './models/resnet50_coco_best_v2.1.0.h5'
		self.model = models.load_model(model_path, backbone_name='resnet50')

		self.threshold = 0.5


	def detect(self, frame):
		# preprocess
		frame = preprocess_image(frame)
		# frame, scale = resize_image(frame)

		# detection
		boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(frame, axis=0))
		# boxes /= scale
		
		bboxes = []
		classIds = []
		confidences = []

		for box, score, label in zip(boxes[0], scores[0], labels[0]):
			# scores are sorted so we can break
			if score < self.threshold:
				break
			x1, y1, x2, y2 = box.astype('int')

			bboxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
			classIds.append(label)
			confidences.append(float(score))
   
		return bboxes, classIds, confidences

if __name__ == '__main__':
	img = cv2.imread('test_image.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	labels_path = './models/coco.names'
	labels_to_names = open(labels_path).read().strip().split('\n')
	retina = Retina()

	boxes, labels, scores = retina.detect(img)

	for box, label, score in zip(boxes, labels, scores):
		print(box, label, score)
		color = label_color(label)
		draw_box(img, box, color=color)
		caption = "{} {:.3f}".format(labels_to_names[label], score)
		draw_caption(img, box, caption)

	cv2.imshow('out', img)
	cv2.waitKey(0)