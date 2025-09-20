import numpy as np
import sklearn
import matplotlib.pyplot as plt
from skimage import io
import scipy
from scipy.signal import convolve2d
import cv2


print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)

# ! - This begins section 1.1 - Convolutions from Scratch!!

# * A convolution is pretty much an element-wise dot product. 
# * We can write a function that takes in a larger image matrix and a smaller convolution kernel 
# * and outputs the resulting matrix 
# * We can also add a padding parameter that pads an image with 0s

example_image = np.arange(1,10).reshape(3,3)


example_kernel = np.full((2,2), 1, dtype=np.int32)



def convolve_2D(image, kernel, pad_width = 0, mul_type="two_loop"):

    image = np.array(image, dtype=np.float32)
    kernel = np.array(kernel, dtype=np.float32)

    # * Here is the padding part of the code, for loops that insert 0s according to pad_width
    # * numpy doesn't allow array size changes so a new array has to be created

    def _pad_image(image, pad_width):
        padded_image = np.full((image.shape[0] + 2 *pad_width, image.shape[1] + 2 * pad_width), 0, dtype=np.float32)

        for row in range(image.shape[0]): 
            for col in range(image.shape[1]):
                padded_image[row + pad_width][col + pad_width] = image[row][col]
            
        return padded_image

    
# * _pad_image is a helper function we can pull out of the bag here only if needed

    if(pad_width):
        image = _pad_image(image, pad_width)

        print(image)

# * After creating the padded image, we can create the holder for the reusult
# * The formula for the shape is padded_image_dim - kernel_dim + 1 

    result = np.full((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1), 0, dtype=np.float32)


# * Since this is a convolution, we should flip the kernel

    kernel = np.flip(kernel)

# * Now we have both the flipped image and the rotated kernel, so we should be able to multiply and add


# * This is a four for-loop convolution implementation
    def element_wise_dot_four(image, kernel):
        for slide_right_num in range(image.shape[0] - kernel.shape[0] + 1):
            for slide_down_num in range(image.shape[1] - kernel.shape[1] + 1):
                addition_holder = 0 
                for row in range(kernel.shape[0]):
                    for col in range(kernel.shape[1]):
                        addition_holder += kernel[row][col] * image[row + slide_right_num][col + slide_down_num]

                result[slide_right_num][slide_down_num] = addition_holder
                addition_holder = 0

        return result

# * This is a two for-loop convolution implementation

    def element_wise_dot_two(image, kernel):
        for row in range(result.shape[0]):
            for col in range(result.shape[1]):
                window = image[row:row+kernel.shape[0], col:col+kernel.shape[1]]

                result[row][col] = np.sum(window * kernel)

        return result


    
    if mul_type == "two_loop":
        result = element_wise_dot_two(image, kernel)
    else:
        result = element_wise_dot_four(image, kernel)

    return result

# ! 1.1 is finished being implemented. I will now compare it with the built-in function

gray = io.imread("./images/bron_dray.jpeg", as_gray=True)

# * Here is a box filter that I will use 
'''
box_filter = 1/9 * np.full((3,3), 1, dtype=np.float32)

my_convolved_image = convolve_2D(gray, box_filter, pad_width=0)

builtin_convolved_image = scipy.signal.convolve2d(gray, box_filter, mode="valid")

# * Plotting both side by side here

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(gray, cmap ="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")


axes[1].imshow(my_convolved_image, cmap="gray")
axes[1].set_title("My Convolution")
axes[1].axis("off")

axes[2].imshow(builtin_convolved_image, cmap="gray")
axes[2].set_title("SciPy Convolution")
axes[2].axis("off")

plt.show()
'''
'''
# * Now I'm going to use these as kernels instead

dx = np.array([[1, 0, -1]])
dy = np.array([[1],[0],[-1]])

my_dx_image = convolve_2D(gray, dx, pad_width=0)

my_dy_image = convolve_2D(gray, dy, pad_width=0)

builtin_dx_image = scipy.signal.convolve2d(gray, dx, mode="valid")
builtin_dy_image = scipy.signal.convolve2d(gray, dy, mode="valid")

# * Plotting both side by side here

fig, axes = plt.subplots(2, 3, figsize=(10, 5))

axes[0][0].imshow(gray, cmap ="gray")
axes[0][0].set_title("Original Image")
axes[0][0].axis("off")


axes[0][1].imshow(my_dx_image, cmap="gray")
axes[0][1].set_title("MY Dx Convolved")
axes[0][1].axis("off")

axes[0][2].imshow(my_dy_image, cmap="gray")
axes[0][2].set_title("MY Dy Convolved")
axes[0][2].axis("off")


axes[1][1].imshow(builtin_dx_image, cmap="gray")
axes[1][1].set_title("Builtin Dx Convolved")
axes[1][1].axis("off")

axes[1][2].imshow(builtin_dy_image, cmap="gray")
axes[1][2].set_title("Builtin Dy Convolved")
axes[1][2].axis("off")


plt.show()
'''

# ! 1.1 is finally over moving on now

# ! Here, I begin part 1.2 - Finite Difference Operator
'''
cameraman = io.imread("./images/cameraman.png", as_gray=True)

dx = np.array([[1, 0, -1]])
dy = np.array([[1],[0],[-1]])

my_dx_image = convolve2d(cameraman, dx, mode="same", boundary="symm")

my_dy_image = convolve2d(cameraman, dy, mode="same", boundary="symm")


grad_mag = np.sqrt(my_dx_image**2 + my_dy_image**2)

threshold = 0.34  # threshold (you may need to tweak this!)

print("Gradient magnitude min:", grad_mag.min())
print("Gradient magnitude max:", grad_mag.max())

edge_image = (grad_mag > threshold).astype(np.uint8)
plt.imshow(edge_image, cmap="gray") 
plt.title(f"Binarized Gradient Magnitude Image (Threshold={threshold})")
plt.axis("off")
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(10, 5))

axes[0].imshow(cameraman, cmap ="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")


axes[1].imshow(my_dx_image, cmap="gray")
axes[1].set_title("dx Convolution")
axes[1].axis("off")

axes[2].imshow(my_dy_image, cmap="gray")
axes[2].set_title("dy Convolution")
axes[2].axis("off")

axes[3].imshow(grad_mag, cmap="gray")
axes[3].set_title("Gradient Magnitude Image")
axes[3].axis("off")

plt.show()
'''

# ! - That section was much easier. Done with 1.2 now

# ! - Moving onto part 1.3 - Derivative of Guassian (DoG) Filter. 

gaussian_1d = cv2.getGaussianKernel(5, 1)   
gaussian_2d = gaussian_1d @ gaussian_1d.T    

cameraman = io.imread("./images/cameraman.png", as_gray=True)

'''
# blur the image with gaussian
blurred = convolve2d(cameraman, gaussian_2d, mode="same", boundary="symm")

dx = np.array([[1, 0, -1]])
dy = np.array([[1],[0],[-1]])

my_dx_image = convolve2d(blurred, dx, mode="same", boundary="symm")
my_dy_image = convolve2d(blurred, dy, mode="same", boundary="symm")

grad_mag = np.sqrt(my_dx_image**2 + my_dy_image**2)

# threshold
threshold = 0.15

edge_image = (grad_mag > threshold).astype(np.uint8)

# plot edge image
plt.imshow(edge_image, cmap="gray") 
plt.title(f"Post-Gaussian Binarized Gradient Magnitude (Threshold={threshold:.2f})")
plt.axis("off")
plt.show()

# plot all steps
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

axes[0].imshow(cameraman, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(blurred, cmap="gray")
axes[1].set_title("Gaussian Blurred")
axes[1].axis("off")

axes[2].imshow(my_dx_image, cmap="gray")
axes[2].set_title("dx Convolution")
axes[2].axis("off")

axes[3].imshow(my_dy_image, cmap="gray")
axes[3].set_title("dy Convolution")
axes[3].axis("off")

axes[4].imshow(grad_mag, cmap="gray")
axes[4].set_title("Gradient Magnitude")
axes[4].axis("off")

plt.show()

'''

#! - It looks like I need a much lower threshold because gaussian blurring reduces all gradients. 
#! It seems like noise shrinks a lot more and real edges shrink a lot less, which has a big effect on the binarized gradient magnitude image.

# * Now we make the DoG Kernel

'''

# Derivative filters
dx = np.array([[1, 0, -1]])
dy = np.array([[1],[0],[-1]])

# Convolve Gaussian with derivative filters → Derivative of Gaussian filters
DoGx = convolve2d(gaussian_2d, dx, mode="same")
DoGy = convolve2d(gaussian_2d, dy, mode="same")

# * Convolve image directly with DoG filters
my_dogx_image = convolve2d(cameraman, DoGx, mode="same", boundary="symm")
my_dogy_image = convolve2d(cameraman, DoGy, mode="same", boundary="symm")

# Gradient magnitude with DoG filters
grad_mag_dog = np.sqrt(my_dogx_image**2 + my_dogy_image**2)

# Threshold (lower because Gaussian smoothing reduces gradients)
threshold_dog = 0.14

edge_image_dog = (grad_mag_dog > threshold_dog).astype(np.uint8)

# Plot edge image
plt.imshow(edge_image_dog, cmap="gray") 
plt.title(f"DoG Filter Binarized Gradient Magnitude (Threshold={threshold_dog:.2f})")
plt.axis("off")
plt.show()

# Plot all steps with DoG
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

axes[0].imshow(cameraman, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(DoGx, cmap="gray")
axes[1].set_title("DoG-x Filter")
axes[1].axis("off")

axes[2].imshow(my_dogx_image, cmap="gray")
axes[2].set_title("DoGx Convolution")
axes[2].axis("off")

axes[3].imshow(my_dogy_image, cmap="gray")
axes[3].set_title("DoGy Convolution")
axes[3].axis("off")

axes[4].imshow(grad_mag_dog, cmap="gray")
axes[4].set_title("DoG Gradient Magnitude")
axes[4].axis("off")

plt.show()

'''

#! - Yea, the dog filter binarized looks pretty much the same!

########### !ONTO PART 2 - FUN WITH FREQUENCIES!  ######################


