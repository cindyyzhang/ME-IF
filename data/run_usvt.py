import numpy as np
import utils.DataMatrix as ME
import utils.USVT as USVT
from utils.utils import *

##################################################################################################
# Hyperparameters
##################################################################################################
etas = [0.01] 
dataset_folder_name = 'samples'
p = [0.1, 0.2]  
cur_mses = []

##################################################################################################
# MAIN
##################################################################################################
if __name__ == "__main__":

	for i in p:
		ratings_ME = ME.DataMatrix()
		processed_data_filename = 'data/' + dataset_folder_name + '/p_' + str(int(i*10))
		ratings_ME.load_from_file(processed_data_filename)
		usvt = USVT.USVT(ratings_ME)
		mses, _ = usvt.run_USVT_and_get_results(etas)
		cur_mses.append(mses[0])
	print(cur_mses)