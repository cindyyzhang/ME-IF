import numpy as np
import utils.DataMatrix as ME
import utils.KNN as KNN
from utils.utils import *

##################################################################################################
# Hyperparameters
##################################################################################################
ks = [10, 20]
dataset_folder_name = 'samples'
ps = [0.1, 0.2, 0.3]   # proportions of data to vary over

##################################################################################################
# MAIN
##################################################################################################
if __name__ == "__main__":

	for i in ps:
		ratings_ME = ME.DataMatrix()
		processed_data_filename = 'data/' + dataset_folder_name + '/p_' + str(int(i*10))
		ratings_ME.load_from_file(processed_data_filename)
		knn = KNN.KNN(ratings_ME)
		mses = knn.run_KNN_and_get_results(ks)