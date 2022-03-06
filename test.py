import os
from google.cloud import storage

"""Stolen from https://medium.com/pradyumn-joshi/how-to-upload-images-on-google-cloud-platform-2a8f594d95d0"""

storage_client = storage.Client.from_service_account_json("key1.json", project="My First Project")
bucket = storage_client.get_bucket("home-cam-one")
path = os.getcwd()
filename = "{}/{}".format(path, "1.jpg")
blob = bucket.blob("images1/{}".format("1.jpg"))
blob.content_type = "image/jpeg"

with open("/dev/shm/06,03,2022 15:18:03.jpg", "rb") as f:
    blob.upload_from_file(f)
    print("Uploaded.")
