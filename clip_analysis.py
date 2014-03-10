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
import subprocess
import argparse

def analyze_clip(video_file, start, end, target, dest, datafile="data",sigma=6, tau=12,descriptor_smoothing=[2,4,8]):
	subprocess.call(["python", "spatiotemporal.py","--sigma",str(sigma),"--tau",str(tau), \
					"--smoothing",str(descriptor_smoothing[0]),str(descriptor_smoothing[1]),\
					str(descriptor_smoothing[2]), "--destination", dest, "--datafile",datafile,\
					video_file,str(start),str(end),target])


def wrapper(x):
	analyze_clip(x[0],x[1],x[2],x[3])


def main():
	parser = argparse.ArgumentParser(description='Analyze spatiotemporal interest points for a set of clips')
	parser.add_argument('inputfile',type=str)
	parser.add_argument('-o','--sigma', metavar='O', type=float,
					   help='spatial convolution scale',default=2.0)
	parser.add_argument('--tau','-t', metavar='T', type=float,
					   help='temporal convolution scale',default=3.0)
	parser.add_argument('--smoothing','-s', type=float,nargs=3,default=[1,2,3])
	parser.add_argument('--fps',type=int,default=30)
	parser.add_argument('--destination','-d',type=str,default="",help="Folder to save results in")
	parser.add_argument('--datafile',type=str,default="data",help="hdf5 file to save descriptors")
	
	sysargs = parser.parse_args()

	# Make directory to store results
	if not sysargs.destination:
		sysargs.destination = "Results-%s-%s" %  (sysargs.inputfile.split('.')[0].split('/')[-1],time.strftime("%d-%m-%y-%H-%M"))

	if os.path.isfile(os.path.join(sysargs.destination,sysargs.datafile+'.hdf5')):
		print "Data file already exists. Exiting..."
		sys.exit()

	if not os.path.isdir(sysargs.destination):	
		print "Creating directory %s..." % sysargs.destination
		os.makedirs(sysargs.destination)
	else:
		print "Saving to %s..." % sysargs.destination

	fps = sysargs.fps
	input_file = sysargs.inputfile
	args = []
	with open(input_file,'r') as fid:
		for line in fid:
			videofile,start,end,target = line.split()
			start = int(start.split(':')[0])*60*fps+int(start.split(':')[1])*fps
			end = int(end.split(':')[0])*60*fps+int(end.split(':')[1])*fps
			args.append((videofile,start,end,target,sysargs.destination))

	with open(os.path.join(sysargs.destination,sysargs.datafile+'-settings.txt'),'w') as fid:
		fid.write("tau: %f\n" % sysargs.tau)
		fid.write("sigma: %f\n" % sysargs.sigma)
		fid.write("smoothing: %f %f %f\n" % tuple(sysargs.smoothing))

	# Don't have the memory to run multiple processes... bummer
	for arg in args:
		print "Analyzing", arg
		analyze_clip(*arg,sigma=sysargs.sigma,tau=sysargs.tau,descriptor_smoothing=sysargs.smoothing,datafile=sysargs.datafile)
		
	print "all done"
	
if __name__ == '__main__':
    main()
