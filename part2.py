import numpy as np
import sklearn
import matplotlib.pyplot as plt
from skimage import io
import scipy
from scipy.signal import convolve2d
import cv2


########### !ONTO PART 2.1 - IMAGE SHARPENING  ######################

#* The point here is to run a gaussian filter over the taj image, which pretty much extracts the low frequencies from the image
#* Once we subtract these low-pass frequencies from the original image, we could be left with the high pass frequencies. This is called the unsharp mask filter.

'''

gaussian_1d = cv2.getGaussianKernel(5, 1)   
gaussian_2d = gaussian_1d @ gaussian_1d.T    

# Load grayscale image
taj = io.imread("./images/taj.jpg", as_gray=True)

# Convolve with Gaussian
taj_blurred = convolve2d(taj, gaussian_2d, mode="same")

taj_subtracted = taj - taj_blurred

'''

def sharpen_image(image):

    gaussian_1d = cv2.getGaussianKernel(5, 1)   
    gaussian_2d = gaussian_1d @ gaussian_1d.T

    image_blurred = convolve2d(image, gaussian_2d, mode="same")

    image_subtracted = image - image_blurred

    image_sharpened = image + image_subtracted

    return image_sharpened

'''

# Plot original and blurred images side by side
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(taj, cmap="gray")
axes[0].set_title("Original Taj")
axes[0].axis("off")

axes[1].imshow(taj_blurred, cmap="gray")
axes[1].set_title("Blurred Taj")
axes[1].axis("off")

axes[2].imshow(taj_subtracted, cmap="gray")
axes[2].set_title("OG - Blurred Taj")
axes[2].axis("off")

taj_sharpened = sharpen_image(taj)

axes[3].imshow(taj_sharpened, cmap="gray")
axes[3].set_title("Sharpened Taj")
axes[3].axis("off")

plt.show()

'''
########### !DONE! Onto part 2.2 - Hybrid Images  ######################

goat = io.imread("./images/goat.jpg", as_gray=True)

lebron = io.imread("./images/lebron.jpg", as_gray=True)

goat = sharpen_image(goat)

image = goat + lebron



plt.imshow(image, cmap="gray")
plt.title("Goat + LeBron")
plt.axis("off")
plt.show()
