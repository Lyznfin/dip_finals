from skimage.color import rgb2gray, rgba2rgb


def to_gray(image):
    if image is None or not image.size:
        raise ValueError("The provided image is empty.")
    
    if len(image.shape) == 2: # Cek apakah citra sudah grayscale
        return image

    try:
        gray_img = rgb2gray(image) # Ubah citra ke format greyscale
    except ValueError:
        rgb_image = rgba2rgb(image) # Convert RGBA (jika citra bertipe rgba) ke RGB
        gray_img = rgb2gray(rgb_image)
    
    return gray_img