import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from geomet import wkt


fig, ax = plt.subplots()


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

def show_plot():
    plt.show()

######################################################################
# Khartoum notes:
# Images are 650 x 650
# Image types are one of PAN (1 band), RGB (3 bands), MUL (8 bands)

# TODO: some im_ids aren't in the dataset e.g. 6
# TODO: some images don't have any polygons e.g. 231
