# spacenet-utils
Utilities for visualizing spacenet data.

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
!(https://github.com/andraugust/spacenet-utils/blob/master/example_output.png?raw=true)
