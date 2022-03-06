import os

import cv2 as cv
import derivative_kernel as dk
import numpy as np
import sys
import math
import collections

import lab2

'''
================================================================================
edge_detector.py
Ripped from myself.
https://github.com/luiszugasti/ele882/blob/master/lab2/lab2_edge_detector.py
Detect  edges  in an image.
Inputs:
    img
        image  being  processed (can be  either  greyscale  or RGB)
    H
        the  filtering  kernel  that  approximates  the  horizontal derivative
    T [optional]
        the  threshold  value  used by the  edge  detector (default  is 0.1) 
    wndsz [optional]
        the  size of the NMS  filter  window (default  is 5)
Outputs:
    edges
        a binary  image  where a value  of  ’1’ indicates  an image  edge
################################################################################
Sample run:
python3 lab2_edge_detector.py /home/luis-zugasti/ele882/lab2/Images/"Section2.2 - Q1"/motion99.512.tiff s 
python3 lab2_edge_detector.py /home/luis-sanroman/Documents/ele882/lab2/Images/"Section2.2 - Q1"/motion99.512.tiff s 
Make sure to uncomment the show image statements, or follow the implied directory structure
================================================================================
'''


def gradient_magnitude(img_a, img_b):
    # assumes imgA and imgB have equal dimensions
    width = img_a.shape[1]
    height = img_b.shape[0]
    img_return = img_a.astype(np.float64)

    for i in range(height):
        for j in range(width):
            img_return[i, j] = math.sqrt(img_a[i, j] ** 2 + img_b[i, j] ** 2)

    return img_return


def or_images(img_a, img_b):
    # assumes imgA and imgB have equal dimensions
    width = img_a.shape[1]
    height = img_b.shape[0]
    img_return = img_a.astype(np.float64)

    for i in range(height):
        for j in range(width):
            if img_a[i, j] == 1 or img_b[i, j] == 1:
                img_return[i, j] = 1

    return img_return


def automatic_threshold(gm):
    width = gm.shape[1]
    height = gm.shape[0]

    derivative_h = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], np.float64)
    x_offset = 1
    y_offset = 1

    add_width = 2
    add_height = 2

    running_derivative = []

    # create area for the mask
    img_to_filter = np.zeros((height + add_height, width + add_width))
    img_to_filter = img_to_filter.astype(np.float64)
    img_to_filter[y_offset:height + y_offset, x_offset:width + x_offset] = gm

    for i in range(x_offset, width + x_offset):
        for j in range(y_offset, height + y_offset):
            derivative_local = 0.0
            # determine filter value at this pixel, dependent on the size of the kernel. Loop through the kernel relative to i, j offsets.
            for k in range(-x_offset, x_offset + 1):
                for el in range(-y_offset, y_offset + 1):
                    derivative_local = derivative_local + derivative_h[el + y_offset, k + x_offset] * img_to_filter[
                        j + el, i + k]
            running_derivative.append(derivative_local)

    # running_derivative.sort()
    # calculate the overall global variance.
    mean = sum(running_derivative) / len(running_derivative)
    variances = []
    for derivative in running_derivative:
        current_variance = (derivative - mean) ** 2
        variances.append(current_variance)
    # now we have a linked list: variances and running_derivative.
    linked_variances = dict(zip(variances, running_derivative))
    linked_variances = collections.OrderedDict(
        sorted(linked_variances.items(), key=lambda t: t[0]))  # first index in the variances, running_derivative items
    # we could employ some form of k-means clustering... with 2 classes: low variance and high variance. But, it may be too close to an NP problem for a simple 882 Lab!
    # so we drill down to the basics. get the top variance (linked to the highest running derivative) and our threshold is half that. Its not perfect.
    highest = linked_variances.popitem(last=True)
    second_highest = linked_variances.popitem(last=True)
    return highest[1] / highest[0]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print("value of T will be automatically determined, wndsz will be set to 5")
        T = None
        wndsz = 5
    elif len(sys.argv) == 5:
        T = float(sys.argv[3])
        wndsz = int(sys.argv[4])
    else:
        print("\n\nusage: python3 edge_detector.py path/to/image H T wndsz \n\
            H: filtering kernel in horizontal. Options:\n\
            cd: Central Difference             [1 0 -1]\n\n\
            fd: Forward Difference             [0 1 -1]\n\n\
            p: Prewitt                         [1 0 -1]\n\
                                            [1 0 -1]\n\
                                            [1 0 -1]\n\n\
            s: Sobel                           [1 0 -1]\n\
                                            [2 0 -2]\n\
                                            [1 0 -1]\n\n\
            T: (optional) threshold value\n\
            wndsz: (optional) size of NMS filter window (one value for height, width)\n\n\
            Default save path of image is os.cwd()\n")
        exit()

    # Open the image. Opencv opens images (grey, black and white, color always in BGR mode. Thus, to get the grey scale we have to threshold it unconditionally)
    image = cv.imread(sys.argv[1])
    lab2.write_image("{}/original_image.png".format(os.getcwd()), image)

    # Convert to grey, and normalize
    image_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_bw_64 = cv.normalize(image_bw.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    lab2.show_image("image", image_bw_64)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_bw_64.png", image_bw_64)

    # Determine Value for derivative kernel
    H = dk.derivative_kernel_return(sys.argv[2])

    # Horizontal Gradient Calculation
    image_bw_64_hg = lab2.spatial_filter(image_bw_64, H)
    H_vertical = H.T  # transpose

    lab2.show_image('horizontal gradient', image_bw_64_hg)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_bw_64_hg.png", image_bw_64_hg)

    # Vertical Gradient Calculation
    image_bw_64_vg = lab2.spatial_filter(image_bw_64, H_vertical)
    lab2.show_image('vertical gradient', image_bw_64_vg)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_bw_64_vg.png", image_bw_64_vg)

    # Gradient Magnitude Calculation
    image_gradient = gradient_magnitude(image_bw_64_hg, image_bw_64_vg)
    lab2.show_image('vertical gradient', image_gradient)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_gradient.png", image_gradient)

    # Horizontal Non-maximum Suppression
    image_h_nms = lab2.non_max_suppress(image_gradient, 1, wndsz)
    lab2.show_image('vertical nms', image_h_nms)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_h_nms.png", image_h_nms)

    # Vertical Non-maximum Suppression
    image_v_nms = lab2.non_max_suppress(image_gradient, wndsz, 1)
    lab2.show_image('horizontal nms', image_v_nms)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_v_nms.png", image_v_nms)

    # did the user set a value for threshold?
    if T is None:
        T = automatic_threshold(image_gradient)
    print("T: {}".format(T))

    # Horizontal Thresholding
    image_tr_h = lab2.image_threshold(image_h_nms, T)
    lab2.show_image('horizontal threshold', image_tr_h)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_tr_h.png", image_tr_h)

    # Vertical Thresholding
    image_tr_v = lab2.image_threshold(image_v_nms, T)
    lab2.show_image('vertical threshold', image_tr_v)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_tr_v.png", image_tr_v)

    # Inclusive OR
    final_edge_map = or_images(image_tr_h, image_tr_v)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/image_tr_v.png", image_tr_v)

    # Output final Image
    lab2.show_image('final edge map', final_edge_map)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result/current_run/final_edge_map.png", final_edge_map)

    '''
    3 problem 2:
    BONUS: Non-linear nature of nms filter.
    '''
    nms_fully_suppressed = lab2.non_max_suppress(image_gradient, 5, 5)
    lab2.show_image('nms_suppressed', nms_fully_suppressed)
    # lab2.write_image("/home/luis-zugasti/ele882/lab2/Images_result2/current_run/nms_fully_suppressed.png",
    #                  nms_fully_suppressed)
    exit()
