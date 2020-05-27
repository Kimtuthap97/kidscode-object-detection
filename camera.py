# import cv2
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from PIL import ImageFont, ImageDraw

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		if not (self.video.isOpened()):
			print("Could not open video device")
		self.yolo = YoloV3Tiny(classes=80)
		self.yolo.load_weights('./weights/yolov3-tiny.tf')
		# print('Hello video')
	def __del__(self):
		self.video.release()
	def get_frame(self):
		font = ImageFont.load_default()
		# vid = cv2.VideoCapture(0)
		ret, frame = self.video.read()
		if frame is None:
			logging.warning("Empty Frame")
		logging.info('weights loaded')
		class_names = [c.strip() for c in open('coco.names').readlines()]
		logging.info('classes loaded')

		img = frame
		img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
		img_in = tf.expand_dims(img_in, 0)
		img_in = transform_images(img_in, 416)
		fps = 0.0
		t1 = time.time()
		boxes, scores, classes, nums = self.yolo.predict(img_in)
		fps  = ( fps + (1./(time.time()-t1)) ) / 2

		img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

		# img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), f1, (0, 0, 255), 2)
		img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)


		ret, jpeg = cv2.imencode('.jpg', img)
		return jpeg.tobytes()