import numpy as np
import spacenet_utils as snu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os
from pprint import pprint as pp
np.random.seed(3)


verbose = True
w = 650     # image width and height
ntr = 18     # number of train images
nte = 1     # number of test images


# define paths to ground truth polygons and images 
summeryData_path = 'AOI_5_Khartoum_Train/summaryData/AOI_5_Khartoum_Train_Building_Solutions.csv'
im_dir = 'AOI_5_Khartoum_Train/MUL-PanSharpen'
im_names = os.listdir(im_dir)
im_paths = [im_dir + '/' + name for name in im_names[1:-1]] # don't include DSstore file

# randomly select training and testing images
tmp = np.random.choice(im_paths,ntr+nte,replace=False)
tr_paths = tmp[0:ntr]
te_paths = tmp[ntr:]

## make training and testing sets
if verbose: print('Making datasets...')
Xtr, ytr = snu.make_dataset(kind='train', im_paths=tr_paths, summeryData_path=summeryData_path)
Xte, yte = snu.make_dataset(kind='test', im_paths=te_paths, summeryData_path=summeryData_path)

## kNN
# train
if verbose: print('Training...')
M = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')
M.fit(Xtr,ytr)
# test
if verbose: print('Testing...')
yM = M.predict(Xte)

## apply filter
if verbose: print('Filtering...')
yMf = snu.postprocess(yM)


## print performance
performance = accuracy_score(yte,yMf)
print('performance = %f' % performance)
C = confusion_matrix(yte, yMf)        # C_ij = number of times labels[i] was predicted to be labels[j]
print('confusion matrix = ')
pp(C)
print('fscore = ' + str(f1_score(yte,yMf)))

## plot
# ground truth
fig, (ax1,ax2) = plt.subplots(1,2)
snu.plot_image(ax1,te_paths[0])
snu.plot_gt(ax1,te_paths[0],summeryData_path)
# predictions
yMf2d = yMf[0:w*w].reshape((w,w))
snu.plot_predictions(ax2,yMf2d)
snu.plot_gt(ax2,te_paths[0],summeryData_path)
plt.show()
