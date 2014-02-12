import sys, os, time
import numpy as np
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
from my_utils import *
import cPickle

class STClip(object):
	def __init__(self,frames):
		self.frames = frames
		self.interest_points = None

	def save(self,filename):
		with open(filename,'w') as fid:
			cPickle.dump(self,fid)

	def load(self,filename):
		with open(filename,'r') as fid:
			clip = cPickle.load(fid)
		return clip

# Capture video
video = cv2.VideoCapture()
success = video.open('../Dataset/person15_handwaving_d1_uncomp.avi')
my_window = cv2.namedWindow('Test')

# Parameters of spatial and temporal range
sigma = 4 # pixels
tau = 16. # frames
clip_length = 300 #frames

f,frame = video.read()
# Full 3-tensor representation of video clip
clip = np.zeros((frame.shape[0],frame.shape[1],clip_length))
blur = np.zeros(clip.shape)

# Skip the first 300 frames
video.set(1,30*0)

# Get a short clip
for i in range(clip_length):
	frame_available, frame = video.read()
	if not frame_available:
		break
	#gs_frame = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(clip.shape[1],clip.shape[0]))
	gs_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	clip[:,:,i] = gs_frame
	#blur[:,:,i] = cv2.Sobel(gs_frame,cv2.CV_32F,1,1)
	blur[:,:,i] = cv2.GaussianBlur(gs_frame,(0,0),sigma)
	
# 1D Gabor filters
t = np.arange(-2*(tau**2),2*(tau**2))
w = 4. / tau
heven= -np.cos(2*np.pi*t*w)*np.exp(-np.square(t)/(tau**2))
hodd = -np.sin(2*np.pi*t*w)*np.exp(-np.square(t)/(tau**2))

# Convolve along time dimension, note that using 'nearest' avoids high
# values of R along the beginning/end of the clip
print "starting even convolution"
even = scipy.ndimage.convolve1d(blur,heven,axis=2,mode='nearest',origin=0)

print "starting odd convolution"
odd = scipy.ndimage.convolve1d(blur,hodd,axis=2,mode='nearest',origin=0)
R = np.square(even) + np.square(odd)

# Center weighting for response function
#enter = clip.shape[1]/2
#width = clip.shape[1]/4
#y = np.arange(clip.shape[1],dtype=float)
#for i in range(clip_length):
#	R[:,:,i] = R[:,:,i] * np.dot(np.ones((clip.shape[0],1)),np.exp(-(y-center)**2/(width**2)).reshape(1,clip.shape[1]))

# Save a video of R
dirname = 'R-%s'%time.strftime("%d-%m-%y-%H-%M")
os.makedirs(dirname)
for i in range(clip_length):
	plt.figure()
	plt.imshow(R[:,:,i])
	plt.savefig(os.path.join(dirname,'frame-%0d.png'%i))
	print "Frame %d, max %f, min %f" % (i, np.max(R[:,:,i]),np.min(R[:,:,i]))

# Now we need to find the local maxima of the response function to identify
# spatial-temporal points of interest
local_max = np.array(detect_local_maxima(R))
print local_max.shape

overlay = np.zeros(clip.shape)
max_x = overlay.shape[0]
max_y = overlay.shape[1]
print "drawing rectangles"
print local_max.shape

#num = 50000
#thresh = 100
# Sort points by descending value of R and take those above 'threshold'
interest_points = sorted([local_max[:,i] for i in range(local_max.shape[1])],key = lambda ip: R[ip[0],ip[1],ip[2]],reverse=True)
R_sorted = np.array([R[ip[0],ip[1],ip[2]] for ip in interest_points])
plt.figure()
plt.hist(R_sorted,bins=50)
plt.savefig(os.path.join(dirname,'local-max-hist.png'))
#cutoff = np.argwhere(R_sorted < thresh)[0]
interest_points = np.array(interest_points).T
print interest_points.shape

# And now sort by time to display in video
interest_points = interest_points[:,np.argsort(interest_points[2,:])]
with open('interest_points.pkl','w') as fid:
	cPickle.dump(interest_points,fid)
	
#stclip = STClip(clip)
#stclip.interest_points = interest_points
#stclip.save('2000pt-repr.pkl')

#for i in range(1,local_max.shape[1],100):
#	x,y,t = local_max[:,i]
#	for tprime in range(max(int(t-tau),0),min(int(t+tau),clip_length)):
#		cv2.rectangle(overlay[:,:,tprime],(x-sigma,x+sigma),(y-sigma,y+sigma),255,thickness=2)

# Let's visualize these interest points
#composite = np.minimum(clip + overlay,255)
#with open('composite.pkl','w') as FILE:
#	cPickle.dump(composite,FILE)
#print "saved"


print clip.shape

tracker = 0
max_x = clip.shape[0]-1
max_y = clip.shape[1]-1
for i in range(clip_length):
	#frame = np.array(clip[:,:,i],dtype=np.uint8)
	while(tracker < interest_points.shape[1] and interest_points[2,tracker] == i):
		x,y,t = interest_points[:,tracker]
		xb,xt = (max(x-sigma,0),min(x+sigma,max_x))
		yb,yt = (max(y-sigma,0),min(y+sigma,max_y))
		for t in range(max(0,i-5),min(clip_length-1,i+5)):
			clip[max(x-sigma,0):min(x+sigma,max_x),max(y-sigma,0):min(y+sigma,max_y),t] = 0
			#print "render rect"
			#print (max(y-sigma,0),max(x-sigma,0)),(min(y+sigma,max_y),min(x+sigma,max_x))
			#frame = clip[:,:,t]
			#cv2.rectangle(frame,(max(x-sigma,0),max(y-sigma,0)),(min(x+sigma,max_x),min(y+sigma,max_y)),0,thickness=2)
			#clip[:,:,t] = frame
		tracker = tracker+1

raw_input("Press Enter to run clip...")
for i in range(clip_length):
	frame = np.array(clip[:,:,i],dtype=np.uint8)
	cv2.imshow("window",frame)
	k = cv2.waitKey(33)
	if k == 1048689: # apparently this is 'q'
		cv2.destroyAllWindows()
		

