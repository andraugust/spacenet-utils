# spacenet-utils
Utilities for modeling and visualizing data from the SpaceNet challenge ().
#
Example usage:

```python
import plot_utils

im_id = '538'       # id of image to plot
csv_path = '/path/to/building-geometries.csv'
im_path = '/path/to/image.tif'

plot_utils.plot_image(im_path,band=1)
plot_utils.plot_gt(im_id,csv_path)
plot_utils.show_plot()
```

Example output:
![alt text](https://github.com/andraugust/spacenet-utils/blob/master/example_output.png?raw=true)

Dependencies: numpy, pandas, osgeo, matplotlib, geomet.
