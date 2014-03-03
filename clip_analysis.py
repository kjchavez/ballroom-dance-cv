"""	Run pipline

	Arguments:
		input_clips:	file with the space-separated fields:
						"video_file start_frame end_frame target_category"
		sigma:			spatial filtering parameter (default = 4)
		tau:			temporal filtering parameter (default = 12)


	Top level script to run the full pipeline:
		1. 	Spatial-temporal interest points and descriptors (multi-threaded)
			a.	Extract local maxima of the response function
			b.	Generate descriptors

		4.	Generate histograms of each of the original input_clips in codeword frequency

	Author: Kevin Chavez
"""

import os,sys
import time
#from multiprocessing import Pool
import subprocess
import argparse

def analyze_clip(video_file, start, end, target, dest, sigma=6, tau=12,descriptor_smoothing=[2,4,8]):
	subprocess.call(["python", "spatiotemporal.py","--sigma",str(sigma),"--tau",str(tau), \
					"--destination", dest, os.path.join('../Dataset',video_file),str(start),str(end),target])

	out_file = video_file.split('/')[-1].split('.')[0]+'-%d-%d'%(start,end)+'.pkl'
	out_file = os.path.join(dest,out_file)

def wrapper(x):
	analyze_clip(x[0],x[1],x[2],x[3])


def main():
	parser = argparse.ArgumentParser(description='Analyze spatiotemporal interest points for a set of clips')
	parser.add_argument('inputfile',type=str)
	parser.add_argument('-o','--sigma', metavar='O', type=int,
					   help='spatial convolution scale',default=2.0)
	parser.add_argument('--tau','-t', metavar='T', type=float,
					   help='temporal convolution scale',default=15.0)
	parser.add_argument('--destination','-d',type=str,default="")
	
	sysargs = parser.parse_args()

	# Make directory to store results
	if not sysargs.destination:
		sysargs.destination = "AnalyzedClips-%s" %  time.strftime("%d-%m-%y-%H-%M")

	if os.path.isdir(sysargs.destination):
		print "Directory already exists. Cannot write data."
		sys.exit()
		
	os.makedirs(sysargs.destination)

	fps = 30
	input_file = sysargs.inputfile
	args = []
	with open(input_file,'r') as fid:
		for line in fid:
			videofile,start,end,target = line.split()
			start = int(start.split(':')[0])*60*fps+int(start.split(':')[1])*fps
			end = int(end.split(':')[0])*60*fps+int(end.split(':')[1])*fps
			args.append((videofile,start,end,target,sysargs.destination))

	with open(os.path.join(sysargs.destination,'settings.txt'),'w') as fid:
		fid.write("tau: %f\n" % sysargs.tau)
		fid.write("sigma: %f\n" % sysargs.sigma)

	# Don't have the memory to run multiple processes... bummer
	for arg in args:
		print "Analyzing", arg
		analyze_clip(*arg,sigma=sysargs.sigma,tau=sysargs.tau)
		
	print "all done"
	
if __name__ == '__main__':
    main()
