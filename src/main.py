import cv2
import numpy as np

from keras_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.keras_retinanet.utils.colors import label_color

from robot import Robot
from yolo import Yolo
from retina import Retina

confThreshold = 0.5
nmsThreshold = 0.4

def draw(yolo, retina, frame):
    boxes, labels, scores = yolo.detect(frame)
    boxes2, labels2, scores2 = retina.detect(frame)
    boxes.extend(boxes2)
    labels.extend(labels2)
    scores.extend(scores2)

    print(boxes)
    print(scores)
    print(labels)
    indices = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

    img = frame.copy()
    for i in indices:
        i = i[0]
        left, top, width, height = boxes[i]
        box = [left, top, left+width, top+height]

        color = label_color(labels[i])
        draw_box(img, box, color=color)

        caption = "{} {:.3f}".format(labels_to_names[labels[i]], scores[i])
        draw_caption(img, box, caption)

    return img

if __name__ == '__main__':
    labels_path = './models/coco.names'
    labels_to_names = open(labels_path).read().strip().split("\n")

    robot = Robot()
    yolo_detector = Yolo()
    retina_detector = Retina()

    robot.start()
    while True:
        frame = robot.getFrame()

        frame = draw(yolo_detector, retina_detector, frame)
        cv2.imshow('ai2thor', frame)

        key = chr(cv2.waitKey(0))
        if key == 'q':
            break
        robot.apply(key)
        

    robot.stop()
    cv2.destroyAllWindows()