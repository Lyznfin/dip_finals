# untuk menambah kontras
from skimage import io, exposure
from .togray import to_gray
from .compare_image import compare_image
from .sharpen import sharpen_image
import numpy as np

# Contrast stretching
# p2, p98 = np.percentile(img, (2, 98))
# img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# Equalization
# img_eq = exposure.equalize_hist(img)

def histogram_equalizer(image):
    grey_image = to_gray(image)
    # histographed = exposure.equalize_adapthist(grey_image, clip_limit=0.03)
    # histographed = exposure.equalize_hist(grey_image)
    p2, p98 = np.percentile(grey_image, (2, 98))
    histographed = exposure.rescale_intensity(grey_image, in_range=(p2, p98))
    return histographed

# if __name__ == '__main__':
#     original = io.imread('image_test/image_cataract_2.png')
#     histog = sharpen_image(original)
#     histog = histogram_equalizer(histog)
#     gray = to_gray(original)
#     compare_image(histog, gray)