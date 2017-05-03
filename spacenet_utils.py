import numpy as np
import pandas as pd
from osgeo import gdal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from geomet import wkt
from PIL import Image, ImageDraw, ImageFilter
import re



def make_dataset(kind='test', im_paths=None, summeryData_path=None):
    '''
    Make dataset of pixel spectra and corresponding ground-truth labels.
    :param kind: 'test' or 'train'.  if 'train', then empty (black) pixels are removed so that kNN is more efficient.
    :param im_paths: list of string paths to geotiff files.
    :param summeryData_path: string path to csv polygon data.  downloads from aws as, e.g., summeryData/AOI_5_Khartoum_Train_Building_Solutions.csv
    :return: Xacc: nparray of spectra for each pixel in each images in im_paths. shape is w*w*len(im_paths) x n_bands
             yacc: nparray of binary ground-truth labels. shape is w*w*len(im_paths). 
    '''
    Xacc = None
    yacc = np.array([])
    for im_path in im_paths:
        # get image pixel array
        X = geotiff2array(im_path)
        w = X.shape[0]              # image width = height
        if Xacc is None:
            n_bands = X.shape[2]
            Xacc = np.empty((0,n_bands))
        ## process polygons
        poly_verts = get_poly_arr(im_path2id(im_path), summeryData_path)
        if not poly_verts:
            y = np.zeros(w*w,dtype=bool)
        else:
            y = poly_verts2mask(poly_verts).flatten().astype(bool)
        X = X.reshape(w*w,n_bands)
        if kind == 'train':
            # remove black pixels
            good_px = (X!=np.zeros(n_bands)).all(axis=1)
            Xacc = np.concatenate((Xacc,X[good_px]))
            yacc = np.concatenate((yacc,y[good_px]))
        elif kind == 'test':
            Xacc = np.concatenate((Xacc,X))
            yacc = np.concatenate((yacc,y))
    return Xacc, yacc


def poly_verts2mask(poly_verts,w=650):
    '''
    convert polygon vertices to ground-truth mask
    :param poly_verts: array of polygon vertices in units of pixels.  each element is an array of the x,y pixel locations.
    :return: w x w binary nparray of ground-truth building pixels. 1 is building, 0 is not-building.
    '''
    img = Image.new('L', (w, w), 0)
    for v in poly_verts:
        vf = v.flatten().tolist()
        ImageDraw.Draw(img).polygon(vf, outline=1, fill=1)
    return np.array(img)


def geotiff2array(im_path):
    '''
    Convert a geotiff image to array of pixel intensities.
    :param im_path: Path to a geotiff image.
    :return: array of pixel intensities. w x w x n_bands
    '''
    imobj = gdal.Open(im_path)
    # extract bands
    nbands = imobj.RasterCount
    if nbands > 1:
        imarr = ()
        for band in range(1,nbands+1):
            imarr += imobj.GetRasterBand(band).ReadAsArray(),
        imarr = np.dstack(imarr)
    else:
        imarr = imobj.GetRasterBand(1).ReadAsArray()
    return imarr


def get_poly_arr(im_id,summeryData_path,im_id_prefix='AOI_5_Khartoum_img'):
    '''
    Get pixel coords of polygon vertices for image with id im_id.  Return as list.
    Requires image ids have prefix given by im_id_prefix.
    :param im_id: string of the integer id of image.
    :param summeryData_path: path to csv file containing polygon vertex information.
    :param im_id_prefix: prefix of field 'image_id' in summaryData csv file.
    :return: array of polygon vertices in units of pixels.
    '''
    # read as dataframe, extract wkt pixels for the desired im_id
    df = pd.read_csv(summeryData_path)
    df = df.loc[df['ImageId']==im_id_prefix+im_id]['PolygonWKT_Pix']
    # convert coords to list
    patches = []
    if df.values[0]!='POLYGON EMPTY':
        # if no ground truth polygons in image, return empty
        for wkt_poly in df:
            lst_poly = np.array(wkt.loads(wkt_poly)['coordinates'][0])
            lst_poly = lst_poly[:,0:2]
            patches.append(lst_poly.astype(int))
    return patches


def plot_image(ax,im_path,band=1):
    '''
    Add satellite image to ax object.
    :param im_path: str path of image file.
    :param band: int band to display (greyscale).
    :return: none
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


def plot_polys(ax,verts):
    '''
    :param ax: pyplot axis object
    :param verts: list of nx2 arrays
    :return: none
    '''
    patches = [Polygon(poly,closed=True) for poly in verts]
    p = PatchCollection(patches, alpha=0.3)
    ax.add_collection(p)


def plot_gt(ax,im_path,summeryData_path,im_id_prefix='AOI_5_Khartoum_img'):
    '''
    Add ground truth polygons to ax object.
    Requires image ids to have prefix given by im_id_prefix.
    :param ax: pyplot axis object
    :param im_id: str id of image
    :param summeryData_path: path to csv file of building geometries
    :param im_id_prefix: prefix of field 'image_id' in summaryData csv file.
    :return: none
    '''
    im_id = im_path2id(im_path)
    # read as dataframe, extract wkt pixels for the desired im_id
    df = pd.read_csv(summeryData_path)
    df = df.loc[df['ImageId']==im_id_prefix+im_id]['PolygonWKT_Pix']       # assumes using AOI_5_Khartoum_img
    if df.values[0]=='POLYGON EMPTY':
        # no ground-truth polygons in image
        return
    # convert coords to list
    patches = []
    for wkt_poly in df:
        lst_poly = np.array(wkt.loads(wkt_poly)['coordinates'][0])
        lst_poly = lst_poly[:,0:2]
        patches.append(Polygon(lst_poly,closed=True))
    # plot
    p = PatchCollection(patches, alpha=0.3)
    ax.add_collection(p)


def plot_predictions(ax,y):
    '''
    :param ax: pyplot axis object
    :param y: w x w
    :return: none
    '''
    ax.imshow(y,cmap='Greys')


def im_path2id(im_path):
    return re.search(r'img(\d*).tif',im_path).group(1)


def postprocess(y, w=650):
    '''
    :param y: Nx1 array of binary model output.  Multiple images are concatenated.
    :param w: image width and height.  assumed the same for all images.
    :return: Nx1 array of filtered output.
    '''
    # convert y to a list of 2d arrays
    nte = int(len(y)/(w*w))       # number of testing images
    y2ds = []                     # will be w x w x nte
    st = 0
    for i in range(nte):
        en = st+w*w
        y2ds.append(y[st:en].reshape((w,w)))
        st = en
    # apply filter
    y2ds = [mode_filter(im) for im in y2ds]
    # convert back to 1d
    return np.concatenate(np.concatenate(y2ds))


def mode_filter(y, width=5, n_app=5):
    '''
    Apply mode filter several times.  Called by postprocess.
    :param width: width of the filter. pixel units. same as height.
    :param y: w x w  nparray of binary values
    :param n_app: number of filter applications
    :return: median filtered nparray w x w
    '''
    yim = Image.fromarray(y).convert('L')
    for i in range(n_app):
        yim = yim.filter(ImageFilter.ModeFilter(size=width))
    return np.array(yim)
