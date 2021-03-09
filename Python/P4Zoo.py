#https://data-flair.training/blogs/python-project-real-time-human-detection-counting/
#https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
#https://realpython.com/face-detection-in-python-using-a-webcam/#the-code

import cv2
import imutils
import numpy as np
import argparse

class peopleCounter:
	def __init__(self):

		self.method = 'face' #appears to be much more reliable, but currently does not have a weight

		#person method
		self.HOGCV = cv2.HOGDescriptor()
		self.HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		self.person = 0
		self.camNumber = 1
		self.width = 640
		self.height = 480
		self.weightLimit = 0.5
		self.weightExclude = 0.1
		self.fontSize = 0.8
		self.padding = (8,8) #Typical values for padding include (8, 8), (16, 16), (24, 24), and (32, 32).
		self.scale = 1.03
		self.winStride = None #(4,4)

		#face method
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #file from https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
		self.scaleFactor = 1.2
		self.minNeighbors = 5
		self.minSize = (30,30)

	def detect(self, frame):
		
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.person = 0

		if (self.method == 'face'):
			faces = self.faceCascade.detectMultiScale(img_gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize,flags=cv2.CASCADE_SCALE_IMAGE)
			# Draw a rectangle around the faces
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				self.person += 1


		else:
			#rects, weights =  self.HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
			
			if (self.winStride is None):
				rects, weights = self.HOGCV.detectMultiScale(img_gray, padding=self.padding, scale=self.scale)
			else: 
				rects, weights = self.HOGCV.detectMultiScale(img_gray, padding=self.padding, scale=self.scale, winStride=self.winStride)


			# for x,y,w,h in rects:
			# 	self.person += 1
			# 	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
			# 	cv2.putText(frame, f'person {self.person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)



			for i, (x, y, w, h) in enumerate(rects):

				if (weights[i] < self.weightExclude):
					continue

				if (weights[i] > self.weightLimit):
					self.person += 1

				if (weights[i] > self.weightExclude and weights[i] < 0.3):
					color = (0,0,255)
				if (weights[i] > 0.3 and weights[i] < 0.7):
					color = (50, 122, 255)
				if (weights[i] > 0.7):
					color = (0, 255, 0)
				cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
				cv2.putText(frame, f'person {self.person}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.5, color, 1)
			cv2.putText(frame, 'High confidence', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.8, (0, 255, 0), 2)
			cv2.putText(frame, 'Moderate confidence', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.8, (50, 122, 255), 2)
			cv2.putText(frame, 'Low confidence', (40, 140), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.8, (0, 0, 255), 2)


		cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,0,255), 2)
		cv2.putText(frame, f'Total Persons : {self.person}', (40,70), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,0,255), 2)

		cv2.imshow('output', frame)

		return frame


	def detectByWebcam(self):   
		video = cv2.VideoCapture(self.camNumber)
		video.set(3, self.width)
		video.set(4, self.height)
		print('Detecting people by camera...')
		while True:
			check, frame = video.read()
			frame = self.detect(frame)
			#cv2.imshow("HOG detection",frame)
			key = cv2.waitKey(1)
			if (key == ord('q')):
				break
		video.release()
		cv2.destroyAllWindows()





if __name__ == "__main__":
	# execute only if run as a script
	test = peopleCounter()
	test.detectByWebcam() #need to hit q to stop