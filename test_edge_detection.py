import cv2 as cv

"""Stolen from https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/image-processing/edge_detection.html"""

# also import the maths module for the square root function
import math

# Open a viewer window to display images
viewer = PIL.ImageShow.Viewer()

# take a picture from the camera
img = PIL.Image.open("{}/{}".format(os.getcwd(), "1.jpg"))

# create an empty image to show the edges
img_edge = PIL.Image(160, 120)

# display the image in the viewer
viewer.displayImage(img)

# iterate over each pixel in the image
for x in range(1, img.width - 1):
    for y in range(1, img.height - 1):
        # initialise Gx to 0 and Gy to 0 for every pixel
        Gx = 0
        Gy = 0

        # top left pixel
        r, g, b = img[x - 1, y - 1]
        # intensity ranges from 0 to 765 (255 * 3)
        intensity = (r + g + b)
        # accumulate the value into Gx, and Gy
        Gx += -intensity
        Gy += -intensity

        # now we do the same for the remaining pixels, left to right, top to bottom

        # remaining left column
        r, g, b = img[x - 1, y]
        Gx += -2 * (r + g + b)

        r, g, b = img[x - 1, y + 1]
        Gx += -(r + g + b)
        Gy += (r + g + b)

        # middle pixels
        r, g, b = img[x, y - 1]
        Gy += -2 * (r + g + b)

        r, g, b = img[x, y + 1]
        Gy += 2 * (r + g + b)

        # right column
        r, g, b = img[x + 1, y - 1]
        Gx += (r + g + b)
        Gy += -(r + g + b)

        r, g, b = img[x + 1, y]
        Gx += 2 * (r + g + b)

        r, g, b = img[x + 1, y + 1]
        Gx += (r + g + b)
        Gy += (r + g + b)

        # calculate the length of the gradient
        length = math.sqrt((Gx * Gx) + (Gy * Gy))

        # normalise the length of gradient to the range 0 to 255
        length = length / 4328 * 255

        # convert the length to an integer
        length = int(length)

        # draw the length in the edge image
        img_edge[x, y] = length, length, length

# display the edge image
viewer.displayImage(img_edge)

# delay for 5000 milliseconds to give us time to see the changes, then exit
time.sleep(5000)

# end of the script
