import cv2
from os import path

cv2_base_dir = path.dirname(path.abspath(cv2.__file__))
haarcascades_dir = cv2_base_dir + "\data\haarcascade_frontalface_default.xml"
print(haarcascades_dir)