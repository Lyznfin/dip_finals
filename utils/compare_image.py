import matplotlib.pyplot as plt
from skimage import io

def compare_image(image1, image2):
    if isinstance(image1, str):
        image1 = io.imread(image1)
    if isinstance(image2, str):
        image2 = io.imread(image2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()