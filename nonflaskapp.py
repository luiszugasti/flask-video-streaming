import datetime
import random
import shutil
import string
import time
from lab2 import write_image

import fs
from fs.memoryfs import MemoryFS

import imutils
from importlib import import_module

import cv2
import cv2 as cv
import os

from google.cloud import storage

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera


def difference_of_images(frame_old, frame_new):
    frame_delta = cv2.absdiff(frame_old, frame_new)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    erroneous_entries = 0
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 200:
            continue
        erroneous_entries = erroneous_entries + 1

    return erroneous_entries > 4


def upload_images(images, image_time_stamps):
    storage_client = storage.Client.from_service_account_json("key1.json", project="My First Project")
    bucket = storage_client.get_bucket("home-cam-one")
    random_folder_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))

    os.mkdir('/dev/shm/{}'.format(random_folder_name))

    for i in range(len(images)):
        cv2.imwrite('/dev/shm/{}/{}.jpg'.format(random_folder_name, image_time_stamps[i]),
                    images[i])
        blob = bucket.blob("images1/{}.jpg".format(image_time_stamps[i]))
        blob.content_type = "image/jpeg"

        result = None
        amt_tries = 0
        sleep_amt = 0.25

        while result is None:

            try:
                blob.upload_from_filename("/dev/shm/{}/{}.jpg".format(random_folder_name, image_time_stamps[i]))
                result = 1
            except:
                # give it up
                if amt_tries > 20:
                    break
                # Just.. try again
                time.sleep(sleep_amt)
                sleep_amt = sleep_amt + 0.25
                pass

    # save some memory
    shutil.rmtree("/dev/shm/{}".format(random_folder_name))


def main_loop():
    MINIMUM_IMAGES = 20

    cam = Camera()
    images = [None] * MINIMUM_IMAGES
    images_bw = [None] * MINIMUM_IMAGES
    timestamps = [None] * MINIMUM_IMAGES
    index = 0
    count = 0
    intruder_detected = False

    while True:
        image = cam.get_frame()

        # Take color image, convert to grayscale, and blur it with gaussian blur
        image_bw_blur = cv.GaussianBlur(cv.cvtColor(image, cv.COLOR_BGR2GRAY), (21, 21), 0)
        timestamp = datetime.datetime.now()

        # Save the image to our list, keep track of the last accessed image
        images[index] = image
        images_bw[index] = image_bw_blur
        timestamps[index] = datetime.datetime.now().strftime("%d,%m,%Y %H:%M:%S")

        index = (index + 1) % MINIMUM_IMAGES

        # if we have exceeded MINIMUM_IMAGES then start comparing the latest image to the oldest image.
        if count < MINIMUM_IMAGES:
            count = count + 1
            continue

        # Check the result of the difference, if it's too high, print something.
        if difference_of_images(images_bw[(index + 1) % MINIMUM_IMAGES], image_bw_blur):
            print("difference omg")
            # upload all images to cloud bucket using a separate thread?
            if not intruder_detected:
                print("id be uploading images now " + datetime.datetime.now().strftime("%d,%m,%Y %H:%M:%S"))
                upload_images(images, timestamps)
            else:
                print("id be uploading one image now" + datetime.datetime.now().strftime("%d,%m,%Y %H:%M:%S"))
                upload_images([image], [timestamps[index]])
            intruder_detected = True
        else:
            intruder_detected = False

        time.sleep(1)


if __name__ == '__main__':
    main_loop()
