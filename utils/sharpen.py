from skimage.filters import laplace
from .togray import to_gray

def sharpen_image(image):
    grey_image = to_gray(image)
    laplacian_image = laplace(grey_image) # Filter laplacian
    sharpened_image = grey_image + laplacian_image # Pertajam citra dengan menambahkan hasil laplace ke image orisinil
    return sharpened_image

# image = 'image_test/image_cataract_2.png'
# sharpened_image = sharpen_image(image)

# compare_image(sharpened_image, image)