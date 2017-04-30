# spacenet-utils
Utilities for modeling and visualizing [SpaceNet](https://crowdsourcing.topcoder.com/spacenet) data.

Dependencies: [numpy](http://www.numpy.org/), [pandas](http://pandas.pydata.org/), [osgeo](https://pypi.python.org/pypi/GDAL), [matplotlib](https://matplotlib.org/), [geomet](https://github.com/geomet/geomet).

### Example usage
Display geotiff image with building-polygon overlay.

```python
# python3
import spacenet_utils as snu
import matplotlib.pyplot as plt

csv_path = 'AOI_5_Khartoum_Train/summaryData/AOI_5_Khartoum_Train_Building_Solutions.csv'
im_path = 'AOI_5_Khartoum_Train/MUL-PanSharpen/MUL-PanSharpen_AOI_5_Khartoum_img5.tif'

fig, ax = plt.subplots()
snu.plot_image(ax,im_path,band=7)
snu.plot_gt(ax,im_path,csv_path)
plt.show()
```
<center><img src="https://github.com/andraugust/spacenet-utils/blob/master/example1.png?raw=true" width="50%"></center>

### Example usage
Label pixels using kNN model.
```bash
$ python example_knn.py
```
![alt text](https://github.com/andraugust/spacenet-utils/blob/master/example2.png?raw=true)
```bash
performance = 0.799041
confusion matrix = 
array([[288411,  13766],
       [ 71139,  49184]])
fscore = 0.536729360026
```


### Note
example_knn.py has only been tested on MUL-PanSharpened images from the Khartoum region.

To use on a different region or image type, be sure to set the following kwargs in spacenet_utils.py: poly_verts2mask.w, postprocess.w, get_poly_arr.im_id_prefix, and plot_gt.im_id_prefix.  See these functions for further documentation.
