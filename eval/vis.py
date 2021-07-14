'''
Evalute + visualize
'''

import itertools

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import numpy as np

def fig_to_arr(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def confmat_to_fig(cm, class_names):
    """
    Taken from:
    https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    # move the class names to the top
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # Normalize the confusion matrix.
    # add 1 to prevent NaNs
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1)
    # reduce precision
    cm_norm = np.around(cm_norm, decimals=2)

    figure = plt.figure(figsize=(16, 16))
    # set min and max values
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Use white text if squares are dark; otherwise black.
    threshold = 0.5
    
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        color = "white" if cm_norm[i, j] > threshold else "black"
        plt.text(j, i, cm_norm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure