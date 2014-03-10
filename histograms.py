"""
	Visualize videos in codeword space and write to file in sparse vector format

"""
import sys, os
import matplotlib.pyplot as plt
import cPickle
import scipy
import h5py

def main():
	folder = sys.argv[1]
	hdf5_file = sys.argv[2]

	# Load pca model
	with open(os.path.join(folder,'pca.pkl'),'r') as fid:
		pca = cPickle.load(fid)

	# Load Kmeans clustering
	with open(os.path.join(folder,'kmeans.pkl'),'r') as fid:
		km = cPickle.load(fid)

	# Create output file for LDA 
	output_file = open(os.path.join(folder,"repr_data.txt"),"w")

	# Load descriptor hdf5 file
	f = h5py.File(hdf5_file,'r')
	descriptors = f['descriptors']

	start = 0
	clip = 1
	for n in descriptors.attrs['lengths']:
		print "Processing clip #%d" % clip
		clip = clip + 1
		# Reduce dimensionality of descriptors
		reduced = pca.transform(descriptors[start:start+n,:])

		# Transform into codeword representation
		clust = km.predict(reduced)
		hist,edges = scipy.histogram(clust,range(km.n_clusters))
		nonzero = scipy.nonzero(hist)[0]

		# Write in sparse format
		output_file.write(str(len(nonzero))+" "+" ".join([str(k)+":"+str(hist[k]) for k in nonzero])+"\n")
		start = start + n
		
	output_file.close()

if __name__ == "__main__":
	main()
	
