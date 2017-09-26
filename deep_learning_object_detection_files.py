# USAGE
# python deep_learning_object_detection_files.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

FindPath = "E:\\JdvrFile\\CapFile\\"
def deleteimage(fullfilename):
	os.remove(fullfilename)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

while True:
	FileNames = os.listdir(FindPath)
	for file_name in FileNames:
		fullfilename = os.path.join(FindPath,file_name)
		# print "fullfilename:", fullfilename
		image = cv2.imread(fullfilename)
		if image is None:
			continue
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0,detections.shape[2]):
			confidence = detections[0,0,i,2]
			if confidence > args["confidence"]:
				idx = int(detections[0,0,i,1])
				box = detections[0,0,i,3:7] * np.array([w,h,w,h])
				(startX,startY,endX,endY) = box.astype("int")
				label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
				cv2.rectangle(image,(startX,startY),(endX,endY),COLORS[idx],2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(image,label,(startX,y),
					cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
		cv2.imshow("Frame",image)
		key = cv2.waitKey(1)
		deleteimage(fullfilename)
