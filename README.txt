************************************************************
*                 CS 231A Final Project
*  
*                Ballroom Dance Recognition
*  
*                      Kevin Chavez
************************************************************

There are a lot of files in here. Let's get a sense of what they
are and what they mean.

THE PIPELINE
------------
1. Run clip_analysis.py to extract spatial-temporal interest points
2. Run codebook.py to generate codebook from subset of descriptors
3. Run histograms.py to convert all clips to codeword histograms
4. Run lda est to estimate the variational parameters of LDA.
5. Run classification.py to print the groups within training data
	that were identified by the LDA
6. To predict a dance category for new clips...
	run predict.py inputfile



DETAILS
----------
clip_analyis.py		

	Takes as input a text file which has the
	following space-separated fields on a line:
	videofile start_time end_time target_style
	i.e. samba.mp4 0:20 0:35 samba

	Outputs: A directory with pickled analyses
	that can be played with play_clip.py
	An hdf5 dataset with descriptors for all
	generated interest points

codebook.py		
	
	Takes as input an hdf5 file, reduced dim for
	descriptor, number of codewords and number of
	videos to use (from the top of list) to 
	create the dictionary.
	Runs PCA and KMeans and saves the models to 
	disk

histograms.py
	
	Input: model_directory hdf5_descriptor_file
	
	Output: repr_data.txt 
	
	This is a sparse vector format
	representation of the videoclips as 
	histograms of codewords. Format is 
	compatible with lda executable

loo.py 

	Perform leave-one-out cross validation
	Input: models_folder, clips_filename, histograms_file, temp_storage_file

lda est			

	See Blei documentation for more details. But
	the command will look something like this:
	./lda est [alpha] [# cat] [settings] [data] random ./lda-model
			
	Outputs: final.* files for the estimated parameters in the 
		lda-model dir.
	
classification.py

	Input:

	Output:

predict.py

	Input:

	Output:
