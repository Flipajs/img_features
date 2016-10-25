# Image Features

Everything you are missing in skimage and other tools

## Color naming features
```python
import img_features
# or any other way how to load image
from scipy.misc import imread


im = imread('path/to/my/img.jpg')

# to obtain feature vector without normalisation
f = img_features.colornames_descriptor(im, block_division=(2, 2), pyramid_levels=2, histogram_density=False)

# with histogram normalisation
f2 = img_features.colornames_descriptor(im, histogram_density=True)


# or if you just want to use basic multilevel feature histogram computation, you can call
feature_data = np.random.rand(300, 200)
img_features.feature_descriptor(feature_data, block_division=(3, 2), pyramid_levels=4, histogram_density=True)


```