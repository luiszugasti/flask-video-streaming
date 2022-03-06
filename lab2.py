import cv2 as cv
import numpy as np
import sys
import math

# /home/luis-sanroman/Documents/ele882/lab2
# /home/luis-zugasti/ele882/lab2
'''
ELE882 lab2: Edge Detection
This laboratory focuses on edge detection on grey scale images.
This library of function serves only for implementing the Edge Detector described in Section 2.2 of the lab file.
'''


# Copy pasta :(
def show_image(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def spatial_filter(img, kernel, *args):
    '''Applies a kernel to an image.
    Can also apply multiple kernels using *args. Note: No checks will be done to verify kernel dimensions are equal.'''
    width = img.shape[1]
    height = img.shape[0]
    width_kernel = kernel.shape[1]
    height_kernel = kernel.shape[0]
    x_offset = (width_kernel - 1) // 2  # force it to be an int
    y_offset = (height_kernel - 1) // 2
    add_width = width_kernel - 1
    add_height = height_kernel - 1

    img_return = img.astype(np.float64)

    mask_return = np.zeros(shape=(1 + len(args), 1), dtype=np.float64)

    # create area for the mask
    img_to_filter = np.zeros(
        shape=(height + add_height, width + add_width),
        dtype=np.float64
    )
    img_to_filter[y_offset:height + y_offset, x_offset:width + x_offset] = img

    # start at the defined offset of the image, in the img_to_filter
    for i in range(x_offset, width + x_offset):
        for j in range(y_offset, height + y_offset):
            # determine filter value at this pixel, dependent on the size of the kernel.
            # Loop through the kernel relative to i, j offsets.
            for k in range(-x_offset, x_offset + 1):
                for l in range(-y_offset, y_offset + 1):
                    mask_return[0, 0] = mask_return[0, 0] + kernel[l + y_offset, k + x_offset] * img_to_filter[j + l, i + k]
                    # do the exact same thing per additional kernel
                    for kernel_index in range(len(args)):
                        mask_return[1 + kernel_index, 0] = mask_return[1 + kernel_index, 0] + \
                                                        kernel[l + y_offset, k + x_offset] * img_to_filter[j + l, i + k]

            # if we got additional kernels, just compute their gradient magnitude
            if len(args) > 0:
                summation = 0.0
                for mask_ret in mask_return:
                    summation = summation + mask_ret ** 2

                img_return[j - y_offset, i - x_offset] = math.sqrt(summation)
            else:
                img_return[j - y_offset, i - x_offset] = mask_return[0, 0]

            mask_return = np.zeros(shape=(1 + len(args), 1), dtype=np.float64)

    return img_return


def non_max_suppress(img, H, W):
    width = img.shape[1]
    height = img.shape[0]
    x_offset = (W - 1) // 2  # force it to be an int
    y_offset = (H - 1) // 2
    add_width = W - 1
    add_height = H - 1
    img_return = img.astype(np.float64)

    # create area for the mask
    img_to_filter = np.zeros((height + add_height, width + add_width))
    img_to_filter.fill(1)
    img_to_filter = img_to_filter.astype(np.float64)
    img_to_filter[y_offset:height + y_offset, x_offset:width + x_offset] = img
    # row,                    column

    for i in range(x_offset, width + x_offset):
        for j in range(y_offset, height + y_offset):
            # within the H, W window: is the current pixel value the greatest? If yes, maintain. Otherwise, zero.
            # build the neighborhood array
            tester = img_to_filter[j - y_offset: j + y_offset + 1, i - x_offset:i + x_offset + 1]
            # if i > 377:
            #     print("i:{} j:{}".format(j, i))
            img_to_filter[j, i]
            if img_to_filter[j, i] < np.amax(tester):
                img_return[j - y_offset, i - x_offset] = 0

    return img_return


def image_threshold(img, T):
    width = img.shape[1]
    height = img.shape[0]
    img_return = img.astype(np.float64)

    for i in range(height):
        for j in range(width):
            if img_return[i, j] < T:
                img_return[i, j] = 0
            else:
                img_return[i, j] = 1

    return img_return


def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    # img = img.astype(np.uint8)
    img = cv.convertScaleAbs(img, alpha=(255.0))
    cv.imwrite(path, img)


# When the script is invoked by its name, instead of an import
if __name__ == "__main__":
    '''
    ================================================================================
    2.1.1 problem 4:
    Test bench when this code is invoked directly.
    '''
    # Fetch images
    nms = cv.imread("/home/luis-zugasti/ele882/lab2/nms-test.png")
    threshold = cv.imread("/home/luis-zugasti/ele882/lab2/threshold-test.png")

    # Convert to grey, and normalize
    nms_bw = cv.cvtColor(nms, cv.COLOR_BGR2GRAY)
    nms_bw_64 = cv.normalize(nms_bw.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    threshold_bw = cv.cvtColor(threshold, cv.COLOR_BGR2GRAY)
    threshold_bw_64 = cv.normalize(threshold_bw.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # Creation of gaussian mask
    h = np.array([[1, 4, 7, 4, 1], \
                  [4, 20, 33, 20, 4], \
                  [7, 33, 55, 33, 7], \
                  [4, 20, 33, 20, 4], \
                  [1, 4, 7, 4, 1]], np.float64) / 331
    # h = np.array([[1, 0, -1],\
    #              [2, 0, -2],\
    #              [1, 0, -1]], np.float64)
    # Test runs

    show_image('nms_bw_64', nms_bw_64)

    '''
    2.1.1 problem 1:
    Implements general filter, where img, h are assumed to be of double type.
    '''
    nms_filtered = spatial_filter(nms_bw_64, h)
    show_image('nms_filtered', nms_filtered)
    write_image("/home/luis-zugasti/ele882/lab2/Images_result2/current_run/nms_filtered.png", nms_filtered)

    '''
    2.1.1 problem 2:
    Implements the NMS filter with parameters H, W specifiying the desired height, width of the filtering window for NMS.
    '''
    nms_suppressed = non_max_suppress(nms_bw_64, 5, 5)
    show_image('nms_suppressed', nms_suppressed)
    write_image("/home/luis-zugasti/ele882/lab2/Images_result2/current_run/nms_suppressed.png", nms_suppressed)

    '''
    2.1.1 problem 3:
    Implements the Thresholding function where img is a double image and T is a double between 0 and 1.
    '''
    threshold_thresholded = image_threshold(threshold_bw_64, 0.25)
    show_image('threshold_0.25', threshold_thresholded)
    write_image("/home/luis-zugasti/ele882/lab2/Images_result2/current_run/threshold_thresholded1.png",
                threshold_thresholded)
    threshold_thresholded = image_threshold(threshold_bw_64, 0.5)
    show_image('threshold_0.5', threshold_thresholded)
    write_image("/home/luis-zugasti/ele882/lab2/Images_result2/current_run/threshold_thresholded2.png",
                threshold_thresholded)
    threshold_thresholded = image_threshold(threshold_bw_64, 0.75)
    show_image('threshold_0.75', threshold_thresholded)
    write_image("/home/luis-zugasti/ele882/lab2/Images_result2/current_run/threshold_thresholded3.png",
                threshold_thresholded)
