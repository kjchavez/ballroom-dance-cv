import sys, os, time
import numpy as np
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
from my_utils import *
import cPickle
import argparse

class ClipST(object):
	def __init__(self,video_file,start_frame,end_frame):
		self.video_file = video_file
		self.start_frame = start_frame
		self.end_frame = end_frame
		self.interest_points = None

	def save(self,filename):
		with open(filename,'w') as fid:
			cPickle.dump(self,fid)

	def load(self,filename):
		with open(filename,'r') as fid:
			clip = cPickle.load(fid)
		return clip

def main():
	parser = argparse.ArgumentParser(description='Collect spatial-temporal interest points for a clip')
	parser.add_argument('filename', metavar='F', type=str,
					   help='video filename (avi, mp4 format)')
	parser.add_argument('start', metavar='S', type=int,
					   help='start frame',default=0)
	parser.add_argument('end', metavar='E', type=int,
					   help='end frame',default=300)
	parser.add_argument('-o','--sigma', metavar='O', type=int,
					   help='spatial convolution scale',default=4)
	parser.add_argument('-t','--tau', metavar='T', type=float,
					   help='temporal convolution scale',default=16.)
	parser.add_argument('-w','--width', metavar='W', type=int,
					   help='force width of video',default=-1)
	parser.add_argument('-g','--height', metavar='H', type=int,
					   help='force height of video',default=-1)
	parser.add_argument('-r','--save-R',dest='save_R',action='store_true',
					   help='save a video of the response function',default=False)				

	args = parser.parse_args()

	# Capture video
	video = cv2.VideoCapture()
	success = video.open(args.filename)
	#my_window = cv2.namedWindow('Test')
	
	# Parameters of spatial and temporal range
	clip_length = args.end - args.start #frames
	assert(clip_length > 0)

	clip_st = ClipST(args.filename,args.start,args.end)
	clip_st.sigma = args.sigma
	clip_st.tau = args.tau

	# Full 3-tensor representation of video clip
	if args.width < 0 and args.height < 0:
		f,frame = video.read()
		clip = np.zeros((frame.shape[0],frame.shape[1],clip_length))
		clip_st.width = frame.shape[0]
		clip_st.height = frame.shape[1]
		
	# Skip to the first frame
	video.set(1,args.start)

	# Load the frames into memory and apply spatial gaussian filter
	for i in range(clip_length):
		frame_available, frame = video.read()
		if not frame_available:
			break

		if args.width > 0 and args.height > 0:
			gs_frame = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(args.height,args.width))
			clip_st.width = args.width
			clip_st.height = args.height
			
		gs_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		clip[:,:,i] = cv2.GaussianBlur(gs_frame,(0,0),args.sigma)
	
	# 1D temporal Gabor filters 
	t = np.arange(-2*(args.tau**2),2*(args.tau**2))
	w = 4. / args.tau
	heven= -np.cos(2*np.pi*t*w)*np.exp(-np.square(t)/(args.tau**2))
	hodd = -np.sin(2*np.pi*t*w)*np.exp(-np.square(t)/(args.tau**2))

	# Convolve along time dimension, note that using 'nearest' avoids high
	# values of R along the beginning/end of the clip
	print "Starting even convolution..."
	even = scipy.ndimage.convolve1d(clip,heven,axis=2,mode='nearest',origin=0)
	print "done."
	print "Starting odd convolution..."
	odd = scipy.ndimage.convolve1d(clip,hodd,axis=2,mode='nearest',origin=0)
	print "done."
	
	# R is the response function for spatial temporal interest. The local
	# maxima of R are used as a sparse representation of the video clip
	R = np.square(even) + np.square(odd)

	# Option to save a video of the response function (for debugging)
	if args.save_R:
		dirname = 'R-%s'%time.strftime("%d-%m-%y-%H-%M")
		os.makedirs(dirname)
		for i in range(clip_length):
			plt.figure()
			plt.imshow(R[:,:,i])
			plt.savefig(os.path.join(dirname,'frame-%0d.png'%i))

	# Now we need to find the local maxima of the response function to identify
	# spatial-temporal points of interest
	interest_points = np.array(detect_local_maxima(R))
	
	# Sort by time for easier display in video
	interest_points = interest_points[:,np.argsort(interest_points[2,:])]

	# Save for future use
	clip_st.interest_points = interest_points
	save_name = args.filename.split('/')[-1].split('.')[0]+'-%d-%d'%(args.start,args.end)+'.pkl'
	clip_st.save(save_name)


if __name__ == "__main__":
	main()
