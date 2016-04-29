import sys
import urllib2
import cv2
import imghdr
import random
import pickle

"""
In this file, we simply take image url as input and extract all the sift descriptors to output.
The data will be further used to generate cluster center of all visual words
"""

sift = cv2.SIFT()

sampleFeatures = list()
counter = 0

for line in sys.stdin:
    #try:
    url = line.strip().split()[0]
    print counter, url
    counter += 1
    with open("image", "wb") as f:
        try:
            f.write(urllib2.urlopen(url, timeout=1).read())
        except:
            continue
    if imghdr.what("image"):
        img = cv2.imread("image")
        kp, descriptor = sift.detectAndCompute(img,None)
        sampleFeatures += list(descriptor)
        
with open("features.pickle", "wb") as f:
    pickle.dump(sampleFeatures, f)

