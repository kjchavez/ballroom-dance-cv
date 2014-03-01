"""
	Visualize videos in codeword space

"""
"""
	# Visualize the clustering
	if args.visualize:
		plt.hist(predicted_codewords,bins=range(km.cluster_centers_.shape[0]))
		plt.show()
"""
import sys, os
import matplotlib.pyplot as plt
import cPickle
import scipy
import scipy.linalg

def main():
	folder = sys.argv[1]
	
	with open(os.path.join(folder,'video_divisions.pkl'),'r') as fid:
		video_divisions = cPickle.load(fid)

	with open(os.path.join(folder,'codeword_membership.pkl'),'r') as fid:
		codeword_membership = cPickle.load(fid)

	num_codewords = max(codeword_membership) + 1
	clips = []
	start = 0
	plt.close('all')
	plt.figure()
	m = int(scipy.sqrt(len(video_divisions)))
	n = len(video_divisions)/m
	for i in range(len(video_divisions)):
		plt.subplot(m,n,i+1)
		clip = codeword_membership[start:start+video_divisions[i]]
		start = start+video_divisions[i]
		clip,bin_edges = scipy.histogram(clip,range(num_codewords+1))
		print clip
		if scipy.linalg.norm(clip) > 0:
			clip = clip / scipy.linalg.norm(clip)
		clips.append(clip)
		
		plt.bar(bin_edges[:-1],clips[-1])
		plt.title('Video %d'%(i+1))
		plt.xlabel('Codeword')
		plt.ylabel('Normalized frequency')

	plt.tight_layout()

	# Plot correlation matrix
	#clips = scipy.concatenate([c.reshape((c.size,1)) for c in clips],axis=1)
	#corr = scipy.dot(clips.T,clips)
	#print corr[:6,:6]
	#print corr[:6,6:]
	#print corr[6:,:6]
	#print corr[6:,6:]
	#plt.figure()
	#plt.imshow(corr,interpolation='nearest')
	#plt.show()

if __name__ == "__main__":
	main()
	
