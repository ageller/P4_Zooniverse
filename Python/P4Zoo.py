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
from numpy.core.multiarray import ndarray
import time
import sys 
import os
import copy

#it appears the openCV does not work with python's multiprocessing, or Threading
from multiprocessing import Process, Manager
manager = Manager()
sharedDict = manager.dict()
sharedDict['frame'] = None
sharedDict['frameDetect'] = None
sharedDict['people'] = None
sharedDict['faceRects'] = None
sharedDict['timerNow'] = 0

#for detecting people and displaying images/buttons
import cv2
import dlib
from imutils import face_utils



#people counter
class peopleDetector():#(QThread):

	def __init__(self, meth='DLIBface', edges=[0,1],):
		#choose between detecting by face or by the full person.  Face seems much more stable
		#options = 'HAARface', 'DNNface', 'DLIBface', 'HOGperson'
		self.method = meth

		#edges to define different regions for detection of people 
		self.edges = edges

		#below are parameters for the different detection methods

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

		#HOGperson method  -- not accurate enough
		#see here for description of options https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
		self.HOGCV = cv2.HOGDescriptor()
		self.HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		self.padding = (8,8) #added pixel padding around image before being sent to classifier.  Typical values for padding include (8, 8), (16, 16), (24, 24), and (32, 32).
		self.scale = 1.03 #scale of image pyramid probably best to stick between 1 and 1.05?
		self.winStride = None #(4,4), step size in x,y for shifting image.  Goes faster with None, but maybe less accurate
		self.weightLimit = 0.5 #only detections with weights above this value will be counted
		self.weightExclude = 0.1 #any detections with weights below this value will be excluded




	def initPeople(self):
		sharedDict['people'] = np.zeros(len(self.edges) - 1, dtype=int)

	def countPeople(self, x):
		people = np.zeros_like(sharedDict['people'])
		if (len([x]) > 0):
			hist, bin_edges = np.histogram([x], bins = self.edges)
			for i,h in enumerate(hist):
				people[i] += int(h)
		sharedDict['people'] = people

	def detect(self):

		try:
			frameDetect = sharedDict['frameDetect']

			if (self.method == 'HAARface'):
				rects = self.faceCascade.detectMultiScale(frameDetect, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize,flags=cv2.CASCADE_SCALE_IMAGE)
				sharedDict['faceRects'] = np.array(rects)

			elif (self.method == 'DNNface'):
				rects = self.dnnFaceDetector(frameDetect, 1)
				rs = []
				for (i, rect) in enumerate(rects):
					x1 = rect.rect.left()
					y1 = rect.rect.top()
					x2 = rect.rect.right()
					y2 = rect.rect.bottom()
					r = [x1, y1, x2 - x1, y2 - y1]
					rs.append(r)
				sharedDict['faceRects'] = np.array(rs)

			elif (self.method == 'DLIBface'):
				rects = self.DLIBFaceDetector(frameDetect, 1)
				rs = []
				for (i, rect) in enumerate(rects):
					(x, y, w, h) = face_utils.rect_to_bb(rect)
					rs.append([x,y,w,h])
				sharedDict['faceRects'] = np.array(rs)

			else: #HOGperson
				
				if (self.winStride is None):
					rects, weights = self.HOGCV.detectMultiScale(frameDetect, padding=self.padding, scale=self.scale)
				else: 
					rects, weights = self.HOGCV.detectMultiScale(frameDetect, padding=self.padding, scale=self.scale, winStride=self.winStride)

				rs = []
				for i, (x, y, w, h) in enumerate(rects):

					if (weights[i] > self.weightLimit):
						rs.append([x,y,w,h])
				sharedDict['faceRects'] = np.array(rs)


			self.countPeople(sharedDict['faceRects'][:,0] + sharedDict['faceRects'][:,2]/2.) #x + w/2.
			#print("here",sharedDict['people'], sharedDict['faceRects'])

		except:
			#print('exception occurred')
			pass


	def run(self): 
		self.initPeople()
		while True:

			#detect people
			#print('called detect', sharedDict['frame'])
			self.detect()




class timer():#(QThread):

	def __init__(self, timerLength=60):

		self.timerLength = timerLength #seconds
		self.startTime = 0
		self.timerNow = self.timerLength 
		self.timerRunning = True #can be toggled with the start/pause button

	def initTimer(self):
		self.startTime = time.time()
		self.timerNow = self.timerLength

	def run(self):
		self.initTimer()
		while True:
			#check the timer
			if (self.timerRunning):
				self.timerNow = self.timerLength - int(np.round(time.time() - self.startTime))

				#I will probalby want to move this control to the main class (?)							
				if (self.timerNow < 0):
					self.initTimer()

				sharedDict['timerNow'] = self.timerNow




class camHandler():
	#using multithreading so that the displayed video is not slowed down by the detection technique
	#all image processing (capture, resize, grayscale, etc.) needs to happen here.  Analysis can happen in a thread.

	def __init__(self, source=0):
		self.source = source

		#define the width and height to use for the displayed image
		self.width = int(640)*2
		self.height = int(480)*2
		self.mainX0 = int(700)
		self.mainY0 = int(0)

		#define the width and height to use in the detection algorithm (<= width and height set above)
		self.detectWidth = int(640)
		self.detectHeight = int(480)

		#button
		self.buttonImagePath = os.path.join('buttonImages','start.png')
		self.buttonWidth = int(200)
		self.buttonHeight = int(200)
		self.buttonX0 = int(1000)
		self.buttonY0 = int(800)

		#edges to define the different regions for people detection
		self.xEdges = [0, 1/3, 2/3, 1]

		#font displayed on top of the video (could define on input)
		self.fontSize = 0.5

		#will hold the camera object
		self.cam = None

		#will hold the detector process
		self.detectorProcess = None

		#will hold the timer process
		self.timer = None
		self.timerLength = 60

	def onClick(self, event, x, y, flags, param): 
		if (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN): 
			print('button clicked')

	def start(self):
		try:
			#set up the camera
			self.cam = cv2.VideoCapture(self.source)
			self.cam.set(3, self.width)
			self.cam.set(4, self.height)
			#in case the width and height are not possible, get the actual image size
			grabbed, frame = self.cam.read()
			self.height, self.width, channels = frame.shape
			cv2.imshow('detector', frame)
			cv2.moveWindow('detector', self.mainX0, self.mainY0)

			#set up the people detector and start that thread
			detectEdges = np.array(self.detectWidth*np.array(self.xEdges), dtype=int)
			d = peopleDetector('DLIBface', detectEdges)
			self.detectorProcess = Process(target=d.run)
			self.detectorProcess.start()

			#set up the timer and start that thread
			t = timer(self.timerLength)
			self.timerProcess = Process(target=t.run)
			self.timerProcess.start()

			#set up the start/pause button
			img = cv2.imread(self.buttonImagePath) 
			cv2.resize(img, (self.buttonWidth, self.buttonHeight))
			cv2.imshow('startButton', img) 
			cv2.moveWindow('startButton', self.buttonX0, self.buttonY0)
			cv2.setMouseCallback('startButton', self.onClick) 


			#define sizes for annotating the image
			edges = np.array(self.width*np.array(self.xEdges), dtype=int)
			xFac = self.width/self.detectWidth
			yFac = self.height/self.detectHeight

			#start the loop to grab frames and add them to the shared dict, and then display the image
			while True:
				grabbed, frame = self.cam.read()
				#print(grabbed, frame)
				if (frame is not None):
					#define the image for the detector (making a copy in case my edits below change this)
					frameResized = cv2.resize(frame, (self.detectWidth, self.detectHeight))
					frameDetect = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
					sharedDict['frameDetect'] = copy.copy(frameDetect)

					#annotate the frame
					if (sharedDict['people'] is not None):
						for i,n in enumerate(sharedDict['people']):
							cv2.line(frame, (edges[i+1] ,0),(edges[i+1] ,self.height),(255,255,255),4)
							cv2.putText(frame, f'Count : {n}', (edges[i]+40,40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,0,255), 2)

					if (sharedDict['faceRects'] is not None):	
						for (x,y,w,h) in sharedDict['faceRects']:
							cv2.rectangle(frame, (int(x*xFac), int(y*yFac)), (int((x + w)*xFac), int((y + h)*yFac)), (0, 255, 0), 3)
					
					if (sharedDict['timerNow'] is not None):	
						cv2.putText(frame, f":{sharedDict['timerNow']}", (40,self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,255, 0), 2)

					sharedDict['frame'] = frame
					print('here', sharedDict['people'])
					cv2.imshow('detector', frame)
					#cv2.waitKey(1)

				if (cv2.waitKey(1) == ord('q')):
					self.stop()
					break

		except KeyboardInterrupt:
			self.stop()

	def stop(self):
		print('done')
		cv2.destroyAllWindows()
		self.detectorProcess.terminate()
		self.detectorProcess.join()
		self.timerProcess.terminate()
		self.timerProcess.join()



if __name__ == "__main__":






	cam = camHandler(1)
	cam.timerLength = 10
	cam.start()


