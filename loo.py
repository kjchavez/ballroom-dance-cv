import sys, os
import subprocess
import numpy as np
import itertools

def main():
	models_folder = sys.argv[1]
	clips_filename = sys.argv[2]
	histograms_filename = sys.argv[3]
	output_filename = sys.argv[4]

	confusion_matrix = np.zeros((6,6))

	for person in range(1,26):
		clips_file = open(clips_filename,'r')
		histograms_file = open(histograms_filename,'r')
		train_filename = "train-"+output_filename
		test_filename = "test-"+output_filename
		train_file = open("train-"+output_filename,'w')
		test_file = open('test-'+output_filename,'w')
		clips = clips_file.readlines()
		histograms = histograms_file.readlines()

		targets = []
		for i in range(len(clips)):
			if "person%02d" % person not in clips[i]:
				 train_file.write(histograms[i])
			else:
				test_file.write(histograms[i])
				targets.append(clips[i].rstrip('\n'))

		clips_file.close()
		histograms_file.close()
		train_file.close()
		test_file.close()

		a = subprocess.call(['3rdParty/lda-c-dist/lda','est','1','6','lda-settings.txt',train_filename, 'random',os.path.join(models_folder,'lda-model-loo-%02d'%person)])

		subprocess.call(['3rdParty/lda-c-dist/lda','inf', 'lda-inf-settings.txt',os.path.join(models_folder,'lda-model-loo-%02d/final'%person),test_filename, os.path.join(models_folder,'lda-model-loo-%02d/test'%person)])

		# Now we have the classification results in...
		results_filename = os.path.join(models_folder,'lda-model-loo-%02d/test-gamma.dat'%person)
		gamma = np.loadtxt(results_filename)
		categories = np.argmax(gamma,axis=1)

		save_filename = os.path.join(models_folder,'lda-model-loo-%02d/classification.dat'%person)
		with open(save_filename,'w') as fd:
			for i in range(max(categories)+1):
				fd.write('\n')
				fd.write("Category %d\n" % i)
				fd.write("------------------\n")
				for k in range(len(categories)):
					if categories[k] == i:
						fd.write(targets[k]+'\n')

		# Calculate errors. Let the following order be established:
		activities = np.array(['running','jogging','walking','boxing','handclapping','handwaving'])
		# Find the ordering that minimizes total number of misclassifications
		indices = [0,1,2,3,4,5]
		min_errors = 10000;
		min_order = indices
		min_cm = np.zeros((6,6))
		for order in itertools.permutations(indices):
			num_errors = 0
			cm = np.zeros((6,6))
			for k in range(len(categories)):
				true_category = [i for i in range(6) if activities[i] in targets[k]][0]
				if activities[order[categories[k]]] not in targets[k]:
					num_errors += 1
				cm[true_category,order[categories[k]]] += 1
					
			if num_errors < min_errors:
				min_errors = num_errors
				min_order = order
				min_cm = cm
				
				
		confusion_matrix += min_cm
		print confusion_matrix
		
	print "FINAL:"
	print confusion_matrix
	np.save(os.path.join(models_folder,'confusion_matrix.npy'),confusion_matrix)
				
if __name__ == "__main__":
	main()
