import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg
import seaborn as sns
from sklearn import metrics
from scipy import interp

def get_confusion_matrix_figure(y_true, y_pred, n_classes, labels=None):
	"""Plot visualization.
	Parameters
	----------
	y_true : sparse labels integers
	y_pred : prediciton probabilities
	n_classes : number dataset classes
	labels=None : name of class labels
	Returns
	-------
	matplotlib.pyplot.Figure
	"""
	if np.ndim(y_pred)>1:
		y_pred=np.argmax(y_pred, axis=-1)

	if labels is None: # get class names
		labels=list(range(n_classes))

	cm=metrics.confusion_matrix(y_true, y_pred)
	cm_df = pd.DataFrame(cm,
					 index = labels,
					 columns = labels)

	fig, ax=plt.subplots(figsize=(6,5))
	ax=sns.heatmap(cm_df, annot=True, fmt="d", linewidths=.5, cmap="Blues")
	ax.set_ylabel('True label')
	ax.set_xlabel('Predicted label')
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5)
	return fig


def get_auc_roc_curve(y_true, y_pred, n_classes, labels=None):
	"""Plot visualization.
	Parameters
	----------
	y_true : sparse labels integers
	y_pred : prediciton probabilities
	n_classes : number dataset classes
	labels=None : name of class labels
	Returns
	-------
	matplotlib.pyplot.Figure
	"""
	y_true=(np.eye(n_classes)[y_true]).astype(np.int32)

	if labels is None: # get class names
		labels=list(range(n_classes))

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
		roc_auc[i] = metrics.auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
	roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

	# Plot all ROC curves
	fig, ax = plt.subplots(figsize=(5,5))
	ax.plot(fpr["micro"], tpr["micro"],
			 label='micro-average ROC curve (area = {0:0.2f})'
				   ''.format(roc_auc["micro"]),
			 color='deeppink', linestyle=':', linewidth=4)

	ax.plot(fpr["macro"], tpr["macro"],
			 label='macro-average ROC curve (area = {0:0.2f})'
				   ''.format(roc_auc["macro"]),
			 color='navy', linestyle=':', linewidth=4)
	lw=2
	for i in range(n_classes):
		ax.plot(fpr[i], tpr[i], lw=lw, label='{} (area = {:0.2f})'.format(labels[i], roc_auc[i]))

	ax.plot([0, 1], [0, 1], 'k--', lw=lw)
	ax.set_xlim([-0.05, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('Receiver operating characteristic (multi-class)')
	ax.legend(loc="lower right")
	return fig
