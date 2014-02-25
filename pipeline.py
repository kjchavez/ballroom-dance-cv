"""	Run pipline

	Arguments:
		input_clips:	file with the space-separated fields:
						"video_file start_frame end_frame target_category"
		sigma:			spatial filtering parameter (default = 4)
		tau:			temporal filtering parameter (default = 12)
		pca_k:
		codebook_size:

	Top level script to run the full pipeline:
		1. 	Spatial-temporal interest points and descriptors (multi-threaded)
			a.	Extract local maxima of the response function
			b.	Generate descriptors

		2. 	Run PCA on the descriptors from all the videos (keep pca_k dimensions)

		3.	Run K-means to generate 'codebook_size' spatial-temporal codewords

		4.	Generate histograms of each of the original input_clips in codeword frequency

	Author: Kevin Chavez
"""

import os,sys
from multiprocessing import Pool
import subprocess

def analyze_clip(video_file, start, end, target,sigma=6, tau=12,descriptor_smoothing=[2,4,8]):
	subprocess.call(["python", "spatiotemporal.py","--sigma",str(sigma),"--tau",str(tau), \
					os.path.join('../Dataset',video_file),str(start),str(end),target])

	out_file = video_file.split('/')[-1].split('.')[0]+'-%d-%d'%(start,end)+'.pkl'
	out_file = os.path.join('AnalyzedClips',out_file)

def wrapper(x):
	analyze_clip(x[0],x[1],x[2],x[3])


def main():
	fps = 30
	input_file = sys.argv[1]
	args = []
	with open(input_file,'r') as fid:
		for line in fid:
			videofile,start,end,target = line.split()
			start = int(start.split(':')[0])*60*fps+int(start.split(':')[1])*fps
			end = int(end.split(':')[0])*60*fps+int(end.split(':')[1])*fps
			args.append((videofile,start,end,target))

	print args
	pool = Pool(processes=4)
	pool.map(wrapper,args)
	print "all done"
	
if __name__ == '__main__':
    main()
