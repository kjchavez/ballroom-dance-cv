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
import sys, os
import cPickle
import argparse
import time

import scipy
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser(description='Generate codebook from set of descriptors')
	parser.add_argument('filename', metavar='F', type=str,
					   help='file that lists the filenames of all analyzed clips')
	parser.add_argument('pca_k', metavar='K', type=int,
					   help='Reduced dimensionality of descriptors',default=100)
	parser.add_argument('codebook_size', metavar='S', type=int,
					   help='How many codewords to create',default=500)
	parser.add_argument('-v','--visualize', dest='visualize', action='store_true',
					   help='generate plots?',default=False)


	args = parser.parse_args()

	path = "/".join(args.filename.split('/')[:-1])
	
	# Create a folder to save results
	save_path = os.path.join(path,"Models-%s"%time.strftime("%d-%m-%y-%H-%M"))
	os.makedirs(save_path)

	descriptors = []
	lengths = []
	print "Loading descriptors..."
	with open(args.filename,'r') as fid:
		for clip_file in fid:
			clip_file = clip_file[:-1]+'-descriptors.pkl'
			with open(os.path.join(path,clip_file),'r') as clip_fid:
				subsampled_descriptor = cPickle.load(clip_fid)[::10,:]
				descriptors.append(subsampled_descriptor)
				lengths.append(subsampled_descriptor.shape[1])

	with open(os.path.join(save_path,'video_divisions.pkl'),'w') as fid:
		cPickle.dump(lengths,fid)	

	print "Stacking all descriptors..."
	descriptors = scipy.concatenate(descriptors,axis=1).T

	print "Running PCA with %d components..." % args.pca_k
	pca = RandomizedPCA(n_components = args.pca_k)
	reduced_descriptors = pca.fit_transform(descriptors)
	del descriptors # to free some memory
	with open(os.path.join(save_path,'reduced_descriptors.pkl'),'w') as fid:
		cPickle.dump(reduced_descriptors,fid)

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
	predicted_codewords = km.fit_predict(reduced_descriptors)
	with open(os.path.join(save_path,'codeword_membership.pkl'),'w') as fid:
		cPickle.dump(predicted_codewords,fid)
		
	print "Clustering Inertia:",km.inertia_
	print "Saving KMeans model..."
	with open(os.path.join(save_path,"kmeans.pkl"),'w') as fid:
		cPickle.dump(km,fid)

if __name__ == "__main__":
	main()
