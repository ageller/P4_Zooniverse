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


 
#class that controls all processes and is called to start app
#https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713
class main(QMainWindow):

	def __init__(self):
		super().__init__()


		#define the webcam number and the width and height (in pixels) to scale the image
		self.camNumber = 1
		self.width = int(640)
		self.height = int(360)

		#divisions
		#should correspond with width; maybe I should make this a fraction?
		self.xEdges = [0, 1/3, 2/3, 1]

		self.setWindowTitle("") 


		#open the start/pause button
		self.button = buttonWindow()
		self.button.show()



	@pyqtSlot(QImage)
	def setImage(self, image):
		self.label.setPixmap(QPixmap.fromImage(image))



	def run(self):
		self.setGeometry(100, 100, self.width, self.height) 

		# create a label (I think this is required to show connect the webcam)
		self.label = QLabel(self)
		self.label.resize(self.width, self.height)

		#set up the timer
		self.timer = timer(self)
		self.timer.init()
		self.timer.timerRunning = True #will use the button later
		self.timer.timerLength = 10 #seconds
		self.timer.start()

		#set up the webcam
		self.cam = webcamCapture(self)
		self.cam.timer = self.timer
		self.cam.camNumber = self.camNumber
		self.cam.width = self.width
		self.cam.height = self.height
		self.cam.xEdges = self.xEdges
		self.cam.changePixmap.connect(self.setImage)
		self.cam.init()
		self.cam.start()

		#set up the opencv2 people counter
		self.counter = countPeople(self)
		self.counter.camNumber = self.camNumber
		self.counter.width = self.width
		self.counter.height = self.height
		self.counter.xEdges = self.xEdges
		self.counter.cam = self.cam
		self.counter.init()
		self.counter.start()

		#show the main window
		self.show()

	def closeEvent(self, event):
		self.button.close()
		event.accept()


class timer(QThread):

	def init(self):
		#timer (start is initialized below)
		self.timerLength = 60 #seconds
		self.startTime = 0
		self.timerNow = self.timerLength 
		self.timerRunning = False #can be toggled with the start/pause button

	def initTimer(self):
		self.startTime = time.time()
		self.timerNow = self.timerLength

	def run(self):
		self.initTimer()
		while True:
			#check the timer
			if (self.timerRunning):
				self.timerNow = self.timerLength - int(np.round(time.time() - self.startTime))


#grab webcam image
class webcamCapture(QThread):

	changePixmap = pyqtSignal(QImage)
	def init(self):
		self.frame = None
		self.people = None

		#font displayed on top of the video (could define on input)
		self.fontSize = 2


	def run(self):
		cap = cv2.VideoCapture(self.camNumber)
		width  = int(cap.get(3))  
		height = int(cap.get(4))  
		self.edges = np.array(width*np.array(self.xEdges), dtype=int)

		while True:

			ret, f = cap.read()
			self.frame = f
			if ret:
				#annotate video
				#timer
				cv2.putText(self.frame, f':{self.timer.timerNow}', (40, height - 40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,255, 0), 4)
				for i in range(len(self.edges) - 1):
				#edge lines
					cv2.line(self.frame, (self.edges[i+1] ,0),(self.edges[i+1] ,height),(255,255,255),8)
					if (self.people is not None):
						#number of people
						cv2.putText(self.frame, f'Count : {self.people[i]}', (self.edges[i]+40,80), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,0,255), 4)


				print(self.people)
				
				#check the timer
				if (self.timer.timerNow < 0): #this will allow the timer to display 0 on the screen
					#for now
					time.sleep(3)

					#reset the timer
					self.timer.initTimer()

				# display the image	
				# https://stackoverflow.com/a/55468544/6622587
				rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				h, w, ch = rgbImage.shape
				bytesPerLine = ch * w
				convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
				p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
				self.changePixmap.emit(p)



		cap.release()
		cv2.destroyAllWindows() #not sure this is still necessary


	def addToImage(self):
		print("foo")

#testing a people counter
class countPeople(QThread):

	changePixmap = pyqtSignal(QImage)

	def init(self):
		#choose between detecting by face or by the full person.  Face seems much more stable
		self.method = 'DLIBface' 

		#define the webcam number and the width and height (in pixels) for the displayed image
		#expecting to have these defined already
		# self.camNumber = 1
		# self.width = int(640)
		# self.height = int(480)


		#define the width and height to use in the detection algorithm (<= width and height set above)
		self.detectWidth = int(640)
		self.detectHeight = int(480)

		# #divisions, not set in input
		# #should correspond with width; maybe I should make this a fraction?
		# self.xEdges = [0, 
		# 			   int(np.round(self.width/3.)), 
		# 			   int(np.round(2.*self.width/3.)), 
		# 			   self.width] 

		self.edges = np.array(self.width*np.array(self.xEdges), dtype=int)


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

		#will store the rects for the faces
		self.faceRects = None

	def initPeople(self):
		self.people = np.zeros(len(self.xEdges) - 1, dtype=int)

	def countPeople(self, x):
		#I think the easiest way is to just hist np.histogram, even though at this point there will be only 1 x value 
		hist, bin_edges = np.histogram([x], bins = self.edges)
		for i,h in enumerate(hist):
			self.people[i] += int(h)


	def detect(self):
		
		try:
			frameResized = cv2.resize(self.cam.frame, (self.detectWidth, self.detectHeight))
			img_gray = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
			self.initPeople()

			if (self.method == 'HAARface'):
				rects = self.faceCascade.detectMultiScale(img_gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize,flags=cv2.CASCADE_SCALE_IMAGE)
				self.faceRects = np.array(rects)

			elif (self.method == 'DNNface'):
				rects = self.dnnFaceDetector(img_gray, 1)
				rs = []
				for (i, rect) in enumerate(rects):
					x1 = rect.rect.left()
					y1 = rect.rect.top()
					x2 = rect.rect.right()
					y2 = rect.rect.bottom()
					r = [x1, y1, x2 - x1, y2 - y1]
					rs.append(r)
				self.faceRects = np.array(rs)

			elif (self.method == 'DLIBface'):
				rects = self.DLIBFaceDetector(img_gray, 1)
				rs = []
				for (i, rect) in enumerate(rects):
					(x, y, w, h) = face_utils.rect_to_bb(rect)
					rs.append([x,y,w,h])
				self.faceRects = np.array(rs)

			else: #HOGperson
				
				if (self.winStride is None):
					rects, weights = self.HOGCV.detectMultiScale(img_gray, padding=self.padding, scale=self.scale)
				else: 
					rects, weights = self.HOGCV.detectMultiScale(img_gray, padding=self.padding, scale=self.scale, winStride=self.winStride)

				rs = []
				for i, (x, y, w, h) in enumerate(rects):

					if (weights[i] > self.weightLimit):
						rs.append([x,y,w,h])
				self.faceRects = np.array(rs)


			self.countPeople(self.faceRects[:,0] + self.faceRects[:,2]/2.) #x + w/2.

			# for i,n in enumerate(self.people):
			# 	cv2.line(frame, (self.xEdges[i+1] ,0),(self.xEdges[i+1] ,self.height),(255,255,255),4)
			# 	cv2.putText(frame, f'Count : {n}', (self.xEdges[i]+40,40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,0,255), 2)

			# cv2.putText(frame, f':{self.timerNow}', (40,self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (255,255, 0), 2)
			# cv2.imshow('output', frame)

			self.cam.people = self.people

		except:
			#print('exception occurred')
			pass




	def run(self): 

		while True:

			#detect people
			self.detect()




#https://www.geeksforgeeks.org/pyqt5-create-circular-push-button/
#start/pause button
class buttonWindow(QWidget): 
	def __init__(self): 
		super().__init__() 

		self.setGeometry(100, 700, 140, 140) 
		self.setWindowFlag(Qt.FramelessWindowHint) 
		self.UiComponents() 
		self.show() 

	# method for creating widgets 
	def UiComponents(self): 

		# creating a push button 
		button = QPushButton("CLICK", self) 
		button.setGeometry(20, 20, 100, 100) 
		button.setStyleSheet("border-radius: 50;  border: 2px solid black") 
		button.clicked.connect(self.clickme) 

	# action method 
	def clickme(self): 
		print("pressed")

if __name__ == "__main__":


	# execute only if run as a script
	# counter = peopleCounter()
	# counter.timerLength = 10
	# counter.timerRunning = True
	# counter.detectByWebcam() #type q to stop

	# create pyqt5 app 
	app = QApplication(sys.argv) 
	w = main()
	w.run()
	sys.exit(app.exec_())
