import numpy as np
import utils.DataMatrix as ME
import matplotlib.pyplot as plt

##################################################################################################
# Hyperparameters
##################################################################################################
clusters = 10   # Number of clusters to sample the user from
users_per_cluster = 100   # Number of users to sample from each cluster
vec_length = 50   # Length of vector sampled for each user
max_val = 0.1   # Maximum value of covariance entries

process_data = True   # True if want to re-process the raw data
percent_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Percentage of the data used to train

dataset_folder_name = 'samples'
#cmap = plt.cm.get_cmap('hsv', clusters)   # Color map for plotting

##################################################################################################
# Helper functions
##################################################################################################
def get_rand_covariance(length, max_val):
	"""
	Generate random nxn diagonal matrix with entries in the range [0, max_val)
	"""
	diagonal = max_val * np.random.rand(length)   # scaled random matrix
	return np.diag(diagonal) 

def sample_split_multivariate(num_clusters, users_per_cluster, length, max_val):
	"""
	Draws users samples from num_clusters clusters
	For each cluster, mean vector of length n is populated with random values from (-1, 1),
	covariance is a random nxn diagonal matrix
	PLOT: 2 dimensions of the generated users, color represents corresponding cluster
	"""
	samples = np.array([]).reshape(num_clusters * users_per_cluster, length)

	for c in range(num_clusters):
		mean = 2 * (np.random.rand(length) - 0.5)
		cov = get_rand_covariance(length = length, max_val=max_val)
		new_samples = np.clip(np.random.multivariate_normal(mean, cov, users_per_cluster), -1, 1)
		samples = np.append(samples, new_samples, axis=0)
	# 	# Uncomment to plot samples
	# 	for s in new_samples:
	# 		plt.plot(s[0], s[1], marker="o", markeredgecolor=cmap(c), markerfacecolor=cmap(c))
	# plt.show()
	return samples

def gen_mask(n_rows, n_columns, p):
	"""
	Create an n_rows x n_columns matrix where proportion p of the entries are 1 and proportion (1-p) are 0
	"""
	return np.random.random((n_rows, n_columns)) <= p  


##################################################################################################
# Main
##################################################################################################

if __name__ == "__main__":
	for p in percent_train:
		data = ME.DataMatrix()
		matrix = sample_split_multivariate(clusters, users_per_cluster, vec_length, max_val)
		mask = gen_mask(clusters * users_per_cluster, vec_length, p)

		if process_data:
			processed_data_filename = 'data/' + dataset_folder_name + '/p_' + str(p)
			data.add_matrix_and_mask(matrix, mask, max_val)
			data.prep_for_ME_methods()
			data.save_to_file(processed_data_filename)
		else:
			data.load_from_file(processed_data_filename)