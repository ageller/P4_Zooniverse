#https://data-flair.training/blogs/python-project-real-time-human-detection-counting/
#https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
#https://realpython.com/face-detection-in-python-using-a-webcam/#the-code
#https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1

#Will need to test this with people of different skin tones


#standard libraries
import numpy as np
from numpy.core.multiarray import ndarray
import time
import sys 
import os

from multiprocessing import Process, Manager
manager = Manager()
sharedDict = manager.dict()
sharedDict['frame'] = None
sharedDict['frameDetect'] = None
sharedDict['people'] = None
sharedDict['faceRects'] = None
sharedDict['timerNow'] = 0
sharedDict['timerRunning'] = False #will control start/pause
sharedDict['timerFinished'] = False #flag for when timer runs down to zoer

#for detecting people and displaying images/buttons
import cv2
import dlib
from imutils import face_utils

from PIL import ImageFont, ImageColor, ImageDraw, Image  

class mainController():
	#using multithreading so that the displayed video is not slowed down by the detection technique
	#all image processing (capture, resize, grayscale, etc.) needs to happen here.  Analysis can happen in a thread.

	def __init__(self, source=0):
		self.source = source

		#self.font = '/Users/ageller/VISUALIZATIONS/Adler_fonts/Heroic/Heroic Condensed/HeroicCondensed-Regular.ttf'
		self.font = '/Users/ageller/VISUALIZATIONS/Adler_fonts/Karla/static/Karla-Regular.ttf'
		
		self.categories = ['Craters','Neither','Spiders']

		self.outputFileName = 'results.csv'
		self.ofile = None #will contain the file object

		#define the width and height to use for the displayed image
		self.camWidth = int(640)*2
		self.camHeight = int(480)*2
		self.camX0 = int(700)
		self.camY0 = int(0)
		self.camFontSize = 60
		self.camFontColor = '#00979D'
		self.camRectColor = '#00979D'
		self.camFadeLength = 40
		self.camFadeIter = 0
		self.fading = False
		self.blankImage = None #will contain an image to fade out to (defined below)

		#define the width and height to use in the detection algorithm (<= width and height set above)
		self.detectWidth = int(640)
		self.detectHeight = int(480)

		#button
		self.buttonImagePath = os.path.join('images','circle.png')
		self.buttonWidth = int(200)
		self.buttonHeight = int(200)
		self.buttonX0 = int(700)
		self.buttonY0 = int(800)
		self.buttonImage = None #will be defined below
		self.buttonFontSize = 60#80
		self.buttonFontColor = '#00979D'

		#edges to define the different regions for people detection
		self.xEdges = [None] #will be defined below [0, 1/3, 2/3, 1]

		#will hold the camera object
		self.cam = None

		#will hold the threaded process
		self.detectorProcess = None
		self.timerProcess = None

		#parameters for timer
		self.timer = None # will contain the timer object
		self.timerLength = 60
		self.timerImagePath = os.path.join('images','circle.png')
		self.timerWidth = int(200)
		self.timerHeight = int(200)
		self.timerX0 = int(1000)
		self.timerY0 = int(800)
		self.timerFontSize = 90#100
		self.timerImg = None #will be defined below

		self.imageIndex = 0 #this is a dummy index for writing to a file, when working with real images, I will use the proper naming convention

	def onClick(self, event, x, y, flags, param): 
		#handle click events in the button window
		if (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN): 
			sharedDict['timerRunning'] = not sharedDict['timerRunning']
			if (sharedDict['timerRunning']):
				sharedDict['timerFinished'] = False
			#print('button clicked', sharedDict['timerRunning'])
			self.toggleButton()

	def toggleButton(self):
		#toggle the text in the start/pause button
		if (sharedDict['timerRunning']):
			text = 'Pause'
		else:
			text = 'Start'
		t = self.buttonImage.copy()
		t = self.addText(t, text, self.font, self.buttonFontSize, color=self.buttonFontColor)
		cv2.imshow('startButton',t)

	def addText(self, img, text, font, fontSize, textX = None, textY = None, color = None):
		#https://www.codesofinterest.com/2017/07/more-fonts-on-opencv.html
		#use PIL to draw the text, since that can use true type fonts
		imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		pilImg = Image.fromarray(imgRGB)  
		draw = ImageDraw.Draw(pilImg)  
		font = ImageFont.truetype(font, fontSize)  
		textsize = font.getsize(text)
		if (textX is None):
			textX = int((img.shape[1] - textsize[0])/2)
		if (textY is None):
			textY = int((img.shape[0] - textsize[1])/2)
		if (color is None):
			color = (0,0,0)
		draw.text((textX, textY), text, color, font=font)  

		return cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)  

	def initStartButton(self):
		#initialize the start/pause button
		img = cv2.imread(self.buttonImagePath) 
		self.buttonImage = cv2.resize(img, (self.buttonWidth, self.buttonHeight))
		t = self.buttonImage.copy()
		text = 'Start'
		t = self.addText(t, text, self.font, self.buttonFontSize, color=self.buttonFontColor)
		cv2.imshow('startButton',t)
		cv2.moveWindow('startButton', self.buttonX0, self.buttonY0)
		cv2.setMouseCallback('startButton', self.onClick) 

	def initTimer(self):
		#initialize the timer and start the process
		img = cv2.imread(self.timerImagePath) 
		self.timerImg = cv2.resize(img, (self.timerWidth, self.timerHeight))
		cv2.imshow('timer', self.timerImg) 
		cv2.moveWindow('timer', self.timerX0, self.timerY0)
		self.timer = timer(self.timerLength)
		self.timer.initTimer()
		self.timerProcess = Process(target=self.timer.run)
		self.timerProcess.start()

	def initDetector(self):
		#initialize the detector and start the process
		detectEdges = np.array(self.detectWidth*np.array(self.xEdges), dtype=int)
		d = peopleDetector('DLIBface', detectEdges)
		self.detectorProcess = Process(target=d.run)
		self.detectorProcess.start()

	def initCam(self):
		#initialize the camera
		self.cam = cv2.VideoCapture(self.source)
		self.cam.set(3, self.camWidth)
		self.cam.set(4, self.camHeight)
		#in case the width and height are not possible, get the actual image size
		grabbed, frame = self.cam.read()
		self.camHeight, self.camWidth, channels = frame.shape
		cv2.imshow('detector', frame)
		cv2.moveWindow('detector', self.camX0, self.camY0)
		#create a blank white image to use for fading out
		self.blankImage = np.zeros(frame.shape,dtype=np.uint8)
		self.blankImage.fill(255)

	def updateTimerDisplay(self, textX, textY):
		#update the text in the display and control what happens when timer ends
		if (sharedDict['timerFinished']):
			sharedDict['timerRunning'] = False
			sharedDict['timerFinished'] = False
			self.toggleButton()
			self.fading = True

		t = self.timerImg.copy()
		text = f":{sharedDict['timerNow']:02}"
		t = self.addText(t, text, self.font, self.timerFontSize, textX=textX, textY=textY, color=self.buttonFontColor)
		cv2.imshow('timer',t)

	def fadeToWhite(self):
		#blend the image with a white frame to fade it out to white (after the time ends)
		self.camFadeIter += 1
		if (self.camFadeIter == self.camFadeLength):
			self.camFadeIter = 0
			self.fading = False

		mix = self.camFadeIter/self.camFadeLength
		return cv2.addWeighted( sharedDict['frame'], 1 - mix, self.blankImage, mix, 0)

	def initOutputFile(self):
		#initialize the output file
		if (os.path.isfile(self.outputFileName)):
			#if the file exists open to append
			self.ofile = open(self.outputFileName, 'a')
		else:
			#if it's a new file, create it and write the header
			self.ofile = open(self.outputFileName, 'w')
			header = 'image'
			for x in self.categories:
				header += ','+x
			self.ofile.write(header+'\n')
			self.ofile.flush()

	def outputResults(self):
		#output the results to a file
		print("results",sharedDict['people'])
		line = f'image{self.imageIndex:04}'
		for p in sharedDict['people']:
			line += ','+str(p)
		self.ofile.write(line+'\n')
		self.ofile.flush()
		self.imageIndex += 1 #this should  be moved to a function that grabs the image (and probably want a better naming convention)

	def start(self):
		#start everything, including the main loop that grabs and displays the camera image

		try:
			#initialize the output file
			self.initOutputFile()

			#set up the camera and define sizes for annotating the image
			self.initCam()
			if (self.xEdges[0] is None):
				self.xEdges = np.linspace(0, 1, len(self.categories) + 1)
			edges = np.array(self.camWidth*np.array(self.xEdges), dtype=int)
			xFac = self.camWidth/self.detectWidth
			yFac = self.camHeight/self.detectHeight
			#get font locations 
			font = ImageFont.truetype(self.font, self.camFontSize) 
			textsize = font.getsize(text = max(self.categories, key=len)+" : 100")
			camTextX = [int((edges[i+1] - edges[i] - textsize[0])/2) + edges[i] for i in range(len(edges) - 1)]

			#set up the people detector and start that thread
			self.initDetector()

			#set up the timer and start that thread
			self.initTimer()
			#get the font location -- in case the numbers are different widths, we don't want the timer jumping around
			img = cv2.imread(self.timerImagePath) 
			img = cv2.resize(img, (self.timerWidth, self.timerHeight))
			font = ImageFont.truetype(self.font, self.timerFontSize)  
			textsize = font.getsize(text = f":{self.timerLength:02}")
			timerTextX = int((img.shape[1] - textsize[0])/2)
			timerTextY = int((img.shape[0] - textsize[1])/2)

			#set up the start/pause button
			self.initStartButton()

			#start the loop to grab frames and add them to the shared dict, and then display the image
			while True:
				if (sharedDict['timerNow'] is not None):	
					self.updateTimerDisplay(timerTextX, timerTextY)

				if (not self.fading):
					grabbed, frame = self.cam.read()
					#print(grabbed, frame)
					if (frame is not None):
						#define the image for the detector (making a copy in case my edits below change this)
						frameResized = cv2.resize(frame, (self.detectWidth, self.detectHeight))
						frameDetect = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
						sharedDict['frameDetect'] = frameDetect.copy()

						#annotate the frame
						if (sharedDict['people'] is not None):
							for i,n in enumerate(sharedDict['people']):
								cv2.line(frame, (edges[i+1] ,0),(edges[i+1] ,self.camHeight),(255,255,255),4)
								text = f'{self.categories[i]} : {n}'
								frame = self.addText(frame, text, self.font, self.camFontSize, textX=camTextX[i], textY=10, color=self.camFontColor)

						if (sharedDict['faceRects'] is not None):	
							for (x,y,w,h) in sharedDict['faceRects']:
								cv2.rectangle(frame, (int(x*xFac), int(y*yFac)), (int((x + w)*xFac), int((y + h)*yFac)), ImageColor.getcolor(self.camRectColor,'RGB')[::-1], 3)
						
						sharedDict['frame'] = frame

				if (self.fading):
					frame = self.fadeToWhite()

				#print('here', sharedDict['people'])
				if (frame is not None):
					cv2.imshow('detector', frame)
				
				#cv2.waitKey(1)
				if (cv2.waitKey(1) == ord('q')):
					self.stop()
					break

				if (sharedDict['timerFinished']):
					self.outputResults()


		except KeyboardInterrupt:
			self.stop()

	def stop(self):
		print('done')
		cv2.destroyAllWindows()
		self.detectorProcess.terminate()
		self.detectorProcess.join()
		self.timerProcess.terminate()
		self.timerProcess.join()
		self.ofile.close()


#people counter
class peopleDetector():

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




class timer():

	def __init__(self, timerLength=60):

		self.timerLength = timerLength #seconds
		self.startTime = 0
		self.timerNow = self.timerLength 

	def initTimer(self):
		self.startTime = time.time()
		self.timerNow = self.timerLength
		sharedDict['timerNow'] = self.timerNow

	def run(self):
		self.initTimer()
		while True:

			#check the timer
			if (sharedDict['timerRunning']):
				self.timerNow = self.timerLength - int(np.round(time.time() - self.startTime))

				#I will probalby want to move this control to the main class (?)							
				if (self.timerNow < 0):
					self.initTimer()
					sharedDict['timerRunning'] = False
					sharedDict['timerFinished'] = True

				sharedDict['timerNow'] = self.timerNow

			else: #when it's paused
				self.startTime = time.time() - (self.timerLength - self.timerNow)



if __name__ == "__main__":

	c = mainController(1)
	c.timerLength = 10
	c.start()


