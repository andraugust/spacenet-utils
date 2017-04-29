import numpy as np
import spacenet_utils as snu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os
from pprint import pprint as pp
######################################################################
# Khartoum notes:
# 1020 images total
# Images are 650 x 650
# fscore is 0.60
np.random.seed(3)

verbose = True
w = 650     # image width and height
ntr = 14     # number of train images
nte = 1     # number of test images


# define paths
summeryData_path = '/Users/andrew/DataMining/Proj/data/AOI_5_Khartoum_Train/summaryData/AOI_5_Khartoum_Train_Building_Solutions.csv'
im_dir = '/Users/andrew/DataMining/Proj/data/AOI_5_Khartoum_Train/MUL-PanSharpen'
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

#np.save('../results/yMf.npy',yMf)

'''
# display first image
fig, ax = plt.subplots()
snu.plot_predictions(ax,yM2ds[0])
plt.show()
'''

# print performance
performance = accuracy_score(yte,yMf)
print('performance = %f' % performance)
C = confusion_matrix(yte, yMf)        # C_ij = number of times labels[i] was predicted to be labels[j]
print('confusion matrix = ')
pp(C)
print('fscore = ' + str(f1_score(yte,yMf)))


'''
# 3 x 2 image plot
f, ((ax00,ax01),(ax10,ax11),(ax20,ax21)) = plt.subplots(3,2)

yM2d_0 = yMf[0:w*w].reshape((w,w))
yM2d_1 = yMf[w*w:2*w*w].reshape((w,w))
yM2d_2 = yMf[2*w*w:3*w*w].reshape((w,w))
## first column
# original image
snu.plot_image(ax00,te_paths[0])
snu.plot_image(ax10,te_paths[1])
snu.plot_image(ax20,te_paths[2])
# add gt
snu.plot_gt(ax00,te_paths[0],summeryData_path)
snu.plot_gt(ax10,te_paths[1],summeryData_path)
snu.plot_gt(ax20,te_paths[2],summeryData_path)
## second column
# predictions
snu.plot_predictions(ax01,yM2d_0)
snu.plot_predictions(ax11,yM2d_1)
snu.plot_predictions(ax21,yM2d_2)
# gt
snu.plot_gt(ax01,te_paths[0],summeryData_path)
snu.plot_gt(ax11,te_paths[1],summeryData_path)
snu.plot_gt(ax21,te_paths[2],summeryData_path)

ax00.set_xticklabels([])
ax00.set_yticklabels([])
ax01.set_xticklabels([])
ax01.set_yticklabels([])
ax10.set_xticklabels([])
ax10.set_yticklabels([])
ax11.set_xticklabels([])
ax11.set_yticklabels([])
ax20.set_xticklabels([])
ax20.set_yticklabels([])
ax21.set_xticklabels([])
ax21.set_yticklabels([])

plt.show()
#plt.savefig('../results/yMf_hires.png',bbox_inches='tight',dpi=600)
'''

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
