import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from geomet import wkt


def plot_image(im_path,band=1):
    '''
    Add satellite image to global ax object.
    :param im_path: String path of image file
    :param band:  Int band to plot
    :return: None
    '''
    imobj = gdal.Open(im_path)
    imarr = imobj.GetRasterBand(band).ReadAsArray()
    ax.imshow(imarr,cmap='gray')
    # extract multiple bands
    #nbands = imobj.RasterCount
    #if nbands > 1:
    #    imarr = ()
    #    for band in range(1,nbands+1):
    #        imarr += imobj.GetRasterBand(band).ReadAsArray(),
    #    imarr = np.dstack(imarr)
    #else:
    #    imarr = imobj.GetRasterBand(1).ReadAsArray()
    # add plot to ax
    #ax.imshow(imarr)


def plot_gt(im_id,csv_path):
    '''
    Add ground truth polygons to global ax object.
    :param im_id: String id of image
    :param csv_path: path to csv file of building geometries
    :return: None
    '''
    # read as dataframe, extract wkt pixels for the desired im_id
    df = pd.read_csv(csv_path)
    df = df.loc[df['ImageId']=='AOI_5_Khartoum_img'+im_id]['PolygonWKT_Pix']       # assumes using AOI_5_Khartoum_img
    # convert coords to list
    patches = []
    for wkt_poly in df:
        lst_poly = np.array(wkt.loads(wkt_poly)['coordinates'][0])
        lst_poly = lst_poly[:,0:2]
        patches.append(Polygon(lst_poly,closed=True))
    # plot
    p = PatchCollection(patches, alpha=0.4)
    ax.add_collection(p)


######################################################################
# total number of Khartoum photos is 1686
# Images are 650 x 650
# Image types are one of PAN (1 band), RGB (3 bands), MUL (8 bands)

# TODO: some integers aren't in the dataset
# TODO: some images don't have any polygons (e.g. 231)
im_id = '538'
photo_type = 'PAN'
csv_path = '/Users/andrew/DataMining/Proj/data/AOI_5_Khartoum_Train/summaryData/AOI_5_Khartoum_Train_Building_Solutions.csv'


# construct image path from image id
root_path = '/Users/andrew/DataMining/Proj/data/AOI_5_Khartoum_Train/'
if photo_type=='PAN':
    im_path =  root_path + 'PAN/PAN_AOI_5_Khartoum_img' + im_id + '.tif'
elif photo_type=='RGB':
    im_path = root_path + 'RGB-PanSharpen/RGB-PanSharpen_AOI_5_Khartoum_img' + im_id + '.tif'
elif photo_type=='MUL':
    im_path = root_path + 'MUL-PanSharpen/MUL-PanSharpen_AOI_5_Khartoum_img' + im_id + '.tif'

# plot
fig, ax = plt.subplots()
plot_image(im_path,band=1)
plot_gt(im_id,csv_path)
plt.xlim((0,650))
plt.ylim((0,650))
plt.show()
