"""
	Script to play a clip with spatial temporal interest points already
	identified.

	Author: Kevin Chavez
"""
import cv2
import numpy as np
from spatiotemporal import ClipAnalysis
import os
import argparse
import cPickle

def main():
	parser = argparse.ArgumentParser(description='Play a clip with interest points')
	parser.add_argument('filename', metavar='F', type=str,
					   help='Pickled analysis ClipST instance')
	parser.add_argument('-o','--output', metavar='O', type=str,
					   help='Directory to save images to (jpg format)',default="")
	args = parser.parse_args()
					   
	with open(args.filename,'r') as fid:
		clip_st = cPickle.load(fid)

	video = cv2.VideoCapture()
	success = video.open(clip_st.video_file)
	video.set(1,clip_st.start_frame)

	# Extend the set of interest points to display rectangles for a longer
	# time
	interest_points = clip_st.interest_points[:,np.argsort(clip_st.interest_points[2,:])]
	clip_length = clip_st.end_frame - clip_st.start_frame 

	print "interest point shape:",interest_points.shape
	clip = np.zeros((clip_st.width,clip_st.height,clip_length))
	print clip.shape
	for i in range(clip_length):
		frame_available,frame = video.read()
		if frame_available:
			clip[:,:,i] = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		else:
			break
			
	tracker = 0
	max_x = clip.shape[0]-1
	max_y = clip.shape[1]-1
	thickness = 2
	clip_st.sigma = int(3*clip_st.sigma)
	print "Drawing rectangles..."
	for i in range(clip_length):
		while(tracker < interest_points.shape[1] and interest_points[2,tracker] == i):
			x,y,t = interest_points[:,tracker]
			xb,xt = (max(x-clip_st.sigma,0),min(x+clip_st.sigma,max_x))
			yb,yt = (max(y-clip_st.sigma,0),min(y+clip_st.sigma,max_y))
			for t in range(max(0,i-0),min(clip_length-1,i+5)):
				clip[xb:(xb+thickness),yb:yt,t] = 255
				clip[xb:xt,yb:(yb+thickness),t] = 255
				clip[(xt-thickness):xt,yb:yt,t] = 255
				clip[xb:xt,(yt-thickness):yt,t] = 255
			tracker = tracker+1

	if args.output:
		os.makedirs(args.output)
		for i in range(clip_length):
			frame = np.array(clip[:,:,i],dtype=np.uint8)
			cv2.imwrite(os.path.join(args.output,"frame-%04d.jpg"%i),frame)

	raw_input("Press Enter to run clip...")
	for i in range(clip_length):
		frame = np.array(clip[:,:,i],dtype=np.uint8)
		cv2.imshow("window",frame)
		k = cv2.waitKey(33)
		if k == 1048689: # apparently this is 'q'
			cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
