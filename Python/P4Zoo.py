#https://data-flair.training/blogs/python-project-real-time-human-detection-counting/
#https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
#https://realpython.com/face-detection-in-python-using-a-webcam/#the-code
#https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1

#Will need to test this with people of different skin tones

#would be nice to have a continuous video going, and just update the numbers when they are available
# - maybe i can run two threads, one for detect and one for display
# - display could be at full resolution?

#standard libraries
import numpy as np
import time

#for detecting people
import cv2
import dlib
from imutils import face_utils


#for the button
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys 

class peopleCounter:
	def __init__(self):

		#choose between detecting by face or by the full person.  Face seems much more stable
		self.method = 'DLIBface' 


		#define the webcam number and the width and height (in pixels) to scale the image
		self.camNumber = 1
		self.width = int(640)
		self.height = int(480)

		#divisions
		#should correspond with width; maybe I should make this a fraction?
		self.xEdges = [0, 
					   int(np.round(self.width/3.)), 
					   int(np.round(2.*self.width/3.)), 
					   self.width] 


		#font displayed on top of the video
		self.fontSize = 0.8

		#HOGperson method  -- not accurate enough
		#see here for description of options https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
		self.HOGCV = cv2.HOGDescriptor()
		self.HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		self.padding = (8,8) #added pixel padding around image before being sent to classifier.  Typical values for padding include (8, 8), (16, 16), (24, 24), and (32, 32).
		self.scale = 1.03 #scale of image pyramid probably best to stick between 1 and 1.05?
		self.winStride = None #(4,4), step size in x,y for shifting image.  Goes faster with None, but maybe less accurate
		self.weightLimit = 0.5 #only detections with weights above this value will be counted
		self.weightExclude = 0.1 #any detections with weights below this value will be excluded

		#HAARface method -- more accurate, fast, but requires face on
		#see here for some info on the options https://realpython.com/face-recognition-with-python/
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #file from https://github.com/opencv/opencv/blob/master/data/haarcascades/
		#self.faceCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
		self.scaleFactor = 1.2 #compensates for distance from camera, probably best to stick between 1 and 1.5?
		self.minNeighbors = 5 #how many objects near current one required to declare a face
		self.minSize = (30,30) #size of each window

		#DNNface method -- best accuracy, but very slow
		#https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
		#https://github.com/davisking/dlib-models/blob/master/mmod_human_face_detector.dat.bz2
		self.dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")


		#DLIBface method -- good accuracy, moderately fast
		#https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
		self.DLIBFaceDetector = dlib.get_frontal_face_detector()

		#will store the number of people (initialized below in initPeople function)
		self.people = None


		#timer (start is initialized below)
		self.timerLength = 60 #seconds
		self.start = 0
		self.timerNow = self.timerLength 
		self.timerRunning = False #can be toggled with the start/pause button

	def initTimer(self):
		self.start = time.time()
		self.timerNow = self.timerLength

	def initPeople(self):
		self.people = np.zeros(len(self.xEdges) - 1, dtype=int)

	def countPeople(self, x):
		#I think the easiest way is to just hist np.histogram, even though at this point there will be only 1 x value 
		hist, bin_edges = np.histogram([x], bins = self.xEdges)
		for i,h in enumerate(hist):
			self.people[i] += int(h)


	def detect(self, frame):
		
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.initPeople()

		if (self.method == 'HAARface'):
			rects = self.faceCascade.detectMultiScale(img_gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize,flags=cv2.CASCADE_SCALE_IMAGE)
			# Draw a rectangle around the faces
			for (x, y, w, h) in rects:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				self.countPeople(x + w/2.)


		elif (self.method == 'DNNface'):
			rects = self.dnnFaceDetector(img_gray, 1)

			for (i, rect) in enumerate(rects):
				x1 = rect.rect.left()
				y1 = rect.rect.top()
				x2 = rect.rect.right()
				y2 = rect.rect.bottom()
				# Rectangle around the face
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				self.countPeople((x1 + x2)/2.)


		elif (self.method == 'DLIBface'):
			rects = self.DLIBFaceDetector(img_gray, 1)

			for (i, rect) in enumerate(rects):
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
				self.countPeople(x + w/2.)

		else: #HOGperson
			
			if (self.winStride is None):
				rects, weights = self.HOGCV.detectMultiScale(img_gray, padding=self.padding, scale=self.scale)
			else: 
				rects, weights = self.HOGCV.detectMultiScale(img_gray, padding=self.padding, scale=self.scale, winStride=self.winStride)



			for i, (x, y, w, h) in enumerate(rects):

				if (weights[i] < self.weightExclude):
					continue

				if (weights[i] > self.weightLimit):
					self.countPeople(x + w/2.)

				if (weights[i] > self.weightExclude and weights[i] < 0.3):
					color = (0,0,255)
				if (weights[i] > 0.3 and weights[i] < 0.7):
					color = (50, 122, 255)
				if (weights[i] > 0.7):
					color = (0, 255, 0)
				cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
				cv2.putText(frame, f'person {self.people[0]}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.5, color, 1)
			cv2.putText(frame, 'High confidence', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.8, (0, 255, 0), 2)
			cv2.putText(frame, 'Moderate confidence', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.8, (50, 122, 255), 2)
			cv2.putText(frame, 'Low confidence', (40, 140), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.8, (0, 0, 255), 2)

		for i,n in enumerate(self.people):
			cv2.line(frame, (self.xEdges[i+1] ,0),(self.xEdges[i+1] ,self.height),(255,255,255),4)
			cv2.putText(frame, f'Count : {n}', (self.xEdges[i]+40,40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,0,255), 2)

		cv2.putText(frame, f':{self.timerNow}', (40,self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,255, 0), 2)
		cv2.imshow('output', frame)

		return frame


	def detectByWebcam(self): 
		# # create pyqt5 app 
		# App = QApplication(sys.argv) 
		  
		# # create the instance of our Window 
		# window = QtWindow() 
		  
		# # start the app 
		# sys.exit(App.exec()) 

		video = cv2.VideoCapture(self.camNumber)
		video.set(3, self.width)
		video.set(4, self.height)
		self.initTimer()
		while True:
			#check for quit
			key = cv2.waitKey(1)
			if (key == ord('q')):
				break


			#check the timer
			if (self.timerRunning):
				self.timerNow = self.timerLength - int(np.round(time.time() - self.start))
			if (self.timerNow < 0): #this will allow the timer to display 0 on the screen
				#for now
				time.sleep(3)

				#update the images

				#save the counts

				#reset the timer
				self.initTimer()


			#get the frame from the webcam and detect the people
			check, frame = video.read()
			frame = self.detect(frame)



		video.release()
		cv2.destroyAllWindows()


#https://www.geeksforgeeks.org/pyqt5-create-circular-push-button/
#for start/pause
class QtWindow(QMainWindow): 
	def __init__(self): 
		super().__init__() 
  
		# setting title 
		self.setWindowTitle("") 
  
		# setting geometry 
		self.setGeometry(100, 100, 600, 400) 
  
		# calling method 
		self.UiComponents() 
  
		# showing all the widgets 
		self.show() 
  
	# method for widgets 
	def UiComponents(self): 
  
		# creating a push button 
		button = QPushButton("CLICK", self) 
  
		# setting geometry of button 
		button.setGeometry(200, 150, 100, 100) 
  
		# setting radius and border 
		button.setStyleSheet("border-radius: 50;  border: 2px solid black") 
  
		# adding action to a button 
		button.clicked.connect(self.clickme) 
  
	# action method 
	def clickme(self): 
  
		# printing pressed 
		print("pressed") 
  


if __name__ == "__main__":
	# execute only if run as a script
	counter = peopleCounter()
	counter.timerLength = 10
	counter.timerRunning = True
	counter.detectByWebcam() #type q to stop