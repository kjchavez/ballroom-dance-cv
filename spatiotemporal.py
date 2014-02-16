"""
	Extracts interesting spatio-temporal points from a video clip and
	generates gradient vector descriptors. These descriptors and other
	meta-information about the source of the video clip are then pickled
	and saved to file.

	Author: Kevin Chavez
"""
import sys, os, time

# For data processing
import numpy as np
import scipy.ndimage
from sklearn.decomposition import PCA
import cv2

# To generate heat maps of the response function
import matplotlib.pyplot as plt

# Other utilities
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

def generate_descriptors(clip,interest_points,delta_x,delta_y,delta_t,smoothing_scales=[2,4,8]):
	descriptors = np.zeros((len(smoothing_scales)*3*(2*delta_x+1)*(2*delta_y+1)*(2*delta_t+1),interest_points.shape[1]))
	for i in xrange(interest_points.shape[1]):
		x,y,t = interest_points[:,i]
		spatial_temporal_cube = clip[x-delta_x:x+delta_x+1,y-delta_y:y+delta_y+1,t-delta_t:t+delta_t+1]
		descriptor = np.zeros((len(smoothing_scales)*3*spatial_temporal_cube.size))
		for n,sigma in enumerate(smoothing_scales):
			# Filter and find gradients
			smoothed_cube = scipy.ndimage.filters.gaussian_filter(spatial_temporal_cube,sigma)
			dx,dy,dt = np.gradient(smoothed_cube)
			
			# The index in the descriptor where the section corresponding to this filter starts
			filter_start_idx = n*3*spatial_temporal_cube.size

			# Assign gradients along each axis to the appropriate dimensions of the descriptor
			descriptor[filter_start_idx+0:filter_start_idx+spatial_temporal_cube.size] = dx.flatten()
			descriptor[filter_start_idx+spatial_temporal_cube.size:filter_start_idx+2*spatial_temporal_cube.size] = dy.flatten()
			descriptor[filter_start_idx+2*spatial_temporal_cube.size:filter_start_idx+3*spatial_temporal_cube.size] = dt.flatten()

		descriptors[:,i] = descriptor

	return descriptors
	
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

def test_descriptors():
	cube = np.random.randint(1,200,(10,10,10))
	interest_points = np.array([[5,5,5],[2,2,2]]).T
	descriptors = generate_descriptors(cube,interest_points,1,1,1)
	
	
if __name__ == "__main__":
	test_descriptors()
