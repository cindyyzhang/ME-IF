import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(mat, show_plots=True, min_val=None,max_val=None):
	if show_plots:
		if min_val is None:
			min_val = mat.min()
			max_val = mat.max()
		ax = sns.heatmap(mat, vmin=min_val, vmax=max_val, cmap=sns.cm.rocket_r)
		plt.show()