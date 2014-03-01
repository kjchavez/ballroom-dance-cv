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

class ClipAnalysis(object):
	""" Results of analysis of a short clip

		Contains the filename of the original video clip, the range of
		frames, the interest points that were identified, and the descriptors
		generated for those interest points, and what parameters were used
		along the way
	"""
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
	parser.add_argument('-o','--sigma', metavar='O', type=int,
					   help='spatial convolution scale',default=4)
	parser.add_argument('--tau','-t', metavar='T', type=float,
					   help='temporal convolution scale',default=16.)
	parser.add_argument('-w','--width', metavar='W', type=int,
					   help='force width of video',default=-1)
	parser.add_argument('-g','--height', metavar='H', type=int,
					   help='force height of video',default=-1)
	parser.add_argument('-r','--save-R',dest='save_R',action='store_true',
					   help='save a video of the response function',default=False)
	parser.add_argument('filename', metavar='F', type=str,
					   help='video filename (avi, mp4 format)')
	parser.add_argument('start', metavar='S', type=int,
					   help='start frame',default=0)
	parser.add_argument('end', metavar='E', type=int,
					   help='end frame',default=300)
	parser.add_argument('target',type=str,help="dance category shown in this clip")				

	args = parser.parse_args()

	# Check if this clip has already been analyzed
	if not os.path.isdir('AnalyzedClips'):
		os.makedirs('AnalyzedClips')
		
	filename = args.filename.split('/')[-1].split('.')[0]+'-%d-%d'%(args.start,args.end)+'.pkl'
	try:
		with open(os.path.join('AnalyzedClips','clips.txt'),'r') as fid:
			if filename+'\n' in fid.readlines():
				run = raw_input("This clip has already been analyzed... Run anyway? (y/n)")
				if run != 'y':
					sys.exit(0)
	except:
		pass
		
	# Capture video
	video = cv2.VideoCapture()
	success = video.open(args.filename)

	clip_length = args.end - args.start #frames
	assert(clip_length > 0)

	clip_analysis = ClipAnalysis(args.filename,args.start,args.end)
	clip_analysis.sigma = args.sigma
	clip_analysis.tau = args.tau

	# Full 3-tensor representation of video clip
	if args.width < 0 and args.height < 0:
		f,frame = video.read()
		clip = np.zeros((frame.shape[0],frame.shape[1],clip_length))
		original_clip = np.zeros((frame.shape[0],frame.shape[1],clip_length))
		clip_analysis.width = frame.shape[0]
		clip_analysis.height = frame.shape[1]
		
	# Skip to the first frame
	video.set(1,args.start)

	# Load the frames into memory and apply spatial gaussian filter
	print "Applying spatial Gaussian blur"
	for i in range(clip_length):
		frame_available, frame = video.read()
		if not frame_available:
			break

		if args.width > 0 and args.height > 0:
			gs_frame = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(args.height,args.width))
			clip_analysis.width = args.width
			clip_analysis.height = args.height
			
		gs_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		clip[:,:,i] = cv2.GaussianBlur(gs_frame,(0,0),args.sigma)
		original_clip[:,:,i] = gs_frame
	
	# 1D temporal Gabor filters 
	t = np.arange(-int(clip_length/4),int(clip_length/4))
	w = 4. / args.tau
	heven= -np.cos(2*np.pi*t*w)*np.exp(-np.square(t)/(args.tau**2))
	hodd = -np.sin(2*np.pi*t*w)*np.exp(-np.square(t)/(args.tau**2))

	# Convolve along time dimension, note that using 'nearest' avoids high
	# values of R along the beginning/end of the clip
	print "Starting even temporal convolution..."
	print "length(kernel) = %d" % len(t)
	even = scipy.ndimage.convolve1d(clip,heven,axis=2,mode='nearest',origin=0)
	print "done."
	print "Starting odd temporal convolution..."
	print "length(kernel) = %d" % len(t)
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
	print "Finding local maxima of response function..."
	interest_points = np.array(detect_local_maxima(R))
	print "Done!"

	# What do we do with points near the boundary of the spatio-temporal cube?
	# For now, ignore them. Shouldn't be critical to the classification
	delta_x = args.sigma
	delta_y = args.sigma
	delta_t = 5

	close_to_boundary_x = scipy.logical_or(interest_points[0,:] - delta_x < 0,interest_points[0,:] + delta_x >= clip.shape[0])
	close_to_boundary_y = scipy.logical_or(interest_points[1,:] - delta_y < 0,interest_points[1,:] + delta_y >= clip.shape[1])
	close_to_boundary_t = scipy.logical_or(interest_points[2,:] - delta_t < 0,interest_points[2,:] + delta_t >= clip.shape[2])
	ignore = scipy.logical_or(scipy.logical_or(close_to_boundary_x,close_to_boundary_y),close_to_boundary_t)
	interest_points = interest_points[:,scipy.logical_not(ignore)]

	print "Found %d interesting spatial-temporal points" % interest_points.shape[1]
	# Save the clip analysis data structure.
	if not os.path.isdir('AnalyzedClips'):
		os.makedirs('AnalyzedClips')
	
	clip_analysis.interest_points = interest_points
	print "Generating descriptors..."
	descriptors = generate_descriptors(original_clip,interest_points,delta_x,delta_y,delta_t)
	filename = args.filename.split('/')[-1].split('.')[0]+'-%d-%d'%(args.start,args.end)
	clip_analysis.save(os.path.join('AnalyzedClips',filename)+".pkl")

	with open(os.path.join('AnalyzedClips',filename+"-descriptors.pkl"),'w') as fid:
		cPickle.dump(descriptors,fid)
		

	# Update the list of analyzed clips
	with open(os.path.join('AnalyzedClips','clips.txt'),'a') as fid:
		fid.write(filename+'\n')
	with open(os.path.join('AnalyzedClips','targets.txt'),'a') as fid:
		fid.write(filename+' '+args.target.lower()+'\n')

	print "...saved."

	
if __name__ == "__main__":
	main()
