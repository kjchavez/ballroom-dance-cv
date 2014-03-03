"""
	Classification
	Run from the same folder with final.gamma
"""
import numpy as np

def main():
	gamma = np.loadtxt('final.gamma')
	categories = np.argmax(gamma,axis=1)
	with open('../targets.txt','r') as fid:
		targets = [line.rstrip('\n') for line in fid]
		
	for i in range(max(categories)):
		print
		print "Category %d" % i
		print "------------------"
		for k in range(len(categories)):
			if categories[k] == i:
				print targets[k]

if __name__ == "__main__":
	main()
