"""
	Generate Codebook

	Author: Kevin Chavez

	Arguments:
		input_file
		pca_k:
		codebook_size:

	Continuing the pipeline:
		2. 	Run PCA on the descriptors from all the videos (keep pca_k dimensions)

		3.	Run K-means to generate 'codebook_size' spatial-temporal codewords
"""
import os
import cPickle
import argparse
import time

import scipy
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import h5py

def main():
	parser = argparse.ArgumentParser(description='Generate codebook from set of descriptors')
	parser.add_argument('filename', metavar='F', type=str,
					   help='file that lists the filenames of all analyzed clips')
	parser.add_argument('pca_k', metavar='K', type=int,
					   help='Reduced dimensionality of descriptors',default=100)
	parser.add_argument('codebook_size', metavar='S', type=int,
					   help='How many codewords to create',default=500)
	parser.add_argument('num_videos',type=int,default=12,help="How many of the videos in the file to use for codebook generation.")
	parser.add_argument('-v','--visualize', dest='visualize', action='store_true',
					   help='generate plots?',default=False)

	parser.add_argument('--pca-model',dest='pca_model',default=None,type=str)


	args = parser.parse_args()

	path = '/'.join(args.filename.split('/')[:-1])

	
	# Create a folder to save results
	save_path = os.path.join(path,"Models-%d-%d"%(args.pca_k,args.codebook_size))
	os.makedirs(save_path)

	lengths = []
	print "Loading descriptors..."
	f = h5py.File(args.filename,'r')
	descriptors = f['descriptors']

	# Choose a subset of descriptors
	end = sum(descriptors.attrs['lengths'][0:args.num_videos])
	descriptors = descriptors[:end:3,:]

	if args.pca_model:
		print "Loading saved PCA model..."
		with open(args.pca_model,'r') as fid:
			pca = cPickle.load(fid)
		reduced_descriptors = pca.transform(descriptors)
	else:
		print "Running PCA with %d components..." % args.pca_k
		pca = RandomizedPCA(n_components = args.pca_k)
		reduced_descriptors = pca.fit_transform(descriptors)
		
	f.close()
	
	print "Saving PCA model..."
	with open(os.path.join(save_path,"pca.pkl"),'w') as fid:
		cPickle.dump(pca,fid)	

	print "Explained variance:", pca.explained_variance_ratio_
	plt.figure()
	s = 0
	cdf = np.zeros((len(pca.explained_variance_ratio_)))
	for i in range(len(pca.explained_variance_ratio_)):
		s = s + pca.explained_variance_ratio_[i]
		cdf[i] = s
	plt.plot(cdf)
	plt.title('PCA Explained Variance')
	plt.xlabel('# components')
	plt.savefig(os.path.join(save_path,'explained_variance.png'))
		
	print "Running KMeans with %d cluster centroids..." %args.codebook_size
	km = KMeans(n_clusters=args.codebook_size)
	km.fit(reduced_descriptors)
	print "Clustering Inertia:",km.inertia_
	print "Saving KMeans model..."
	with open(os.path.join(save_path,"kmeans.pkl"),'w') as fid:
		cPickle.dump(km,fid)

if __name__ == "__main__":
	main()
