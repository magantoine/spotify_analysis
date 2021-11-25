import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from functools import reduce
import scipy.stats as stats
import scipy
from astropy.stats import bootstrap
import pandas as pd

def plot(X,Y,clf,show=True,dataOnly=False):
    
    plt.figure()
    # plot data points
    X1 = X[Y==0]
    X2 = X[Y==1]
    Y1 = Y[Y==0]
    Y2 = Y[Y==1]
    class1 = plt.scatter(X1[:, 0], X1[:, 1], zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)
    class2 = plt.scatter(X2[:, 0], X2[:, 1], zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)
    if not dataOnly:
        # get the range of data
        x_min = X[:, 0].min() 
        x_max = X[:, 0].max() 
        y_min = X[:, 1].min() 
        y_max = X[:, 1].max() 

        # sample the data space
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

        # apply the model for each point
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)

        # plot the partitioned space
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        
        # plot hyperplanes
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-1, 0, 1], alpha=0.5)
        
        # plot support vectors
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                edgecolors='g', s=100, linewidth=1)
    if dataOnly:
        plt.title('Data Set')
    else:
        if clf.kernel == 'rbf':
            plt.title('Decision Boundary and Margins, C={}, gamma={} with {} kernel'.format(clf.C,clf.gamma, clf.kernel)) 
        elif clf.kernel == 'poly':
            plt.title('Decision Boundary and Margins, C={}, degree={} with {} kernel'.format(clf.C,clf.degree, clf.kernel)) 
        else:
            plt.title('Decision Boundary and Margins, C={} with {} kernel'.format(clf.C, clf.kernel)) 
        
    plt.legend((class1,class2),('Claas A','Class B'),scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8)
    if show:
        plt.show()

        

def CIs(data, columns, funcs, interval=(2.5, 97.5)):
    """
    computes 95% confidence interval for any column and inputed statistics
    
    input :
        - data : the dataframe on which you want to work
        - columns : list of the names of the columns of the dataframe you want to study
        - funcs : list of the statistics you compute as list [callable function]
    
    returns :
        - a dataframe containing for each column, the low value and the high value (in two difference lines)
        composing the confidence interval
    """
    low, high = interval
    cols = {}
    named_func = [[func.__name__, func] for func in funcs]
    for feature in columns :
        col = np.array([])
        studied_vals = data[feature].dropna(how='any').values
        for func_name, func in named_func:
            boots = bootstrap(studied_vals, bootfunc=func)
            ci = [np.nanpercentile(boots, low),np.nanpercentile(boots, high)]
            col = np.append(np.append(np.append(col, ci[0]), func(studied_vals)), ci[1])
        cols[feature] = col
    index = reduce(lambda x, y : x + y, [[f"{func_name}_low", f"{func_name}_computed", f"{func_name}_high"] for func_name, func in named_func])
    CI_df = pd.DataFrame.from_dict(cols) 
    CI_df.index = index
    return CI_df
