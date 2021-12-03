import cv2
import csv
import numpy as np
import pandas as pd
import math
import json
import matplotlib.pyplot as plt
import time


def entropy(pos):
    if pos == 0 or pos == 1:
        return 0
    return -1 * pos * math.log2(pos) - 1 * (1 - pos) * math.log2(1 - pos)

datasample = pd.read_csv('sample.csv')
linkdata = datasample['link']

numberlink = []
numberlinks = []


for a in linkdata:
    numberlink = []
    stringarray = a[1:-1]
    stringarray = stringarray.split('[')
    for b in stringarray:
        first = b.find(']')
        linkarray = b[0:first]
        linkarray = linkarray.split(',')
        if(linkarray[0] != ''):
            linktype = linkarray[0][1:-1]
            linkleft = int(linkarray[1].strip())
            linkright = int(linkarray[2].strip())
            if(linktype == 'number'):
                numberlink.append([linktype, linkleft, linkright])
    numberlinks.append(numberlink)
index = 0
current_state = []
nforeLink_pos = []
nbackLink_pos = []
nforeLink_entp = []
nbackLink_entp = []
for links in numberlinks:
    maxlink = 1

    nforeLink_pos = [0 for i in range(index)]
    nbackLink_pos = [0 for i in range(index)]
    nforeLink_entp = [0 for i in range(index)]
    nbackLink_entp = [0 for i in range(index)]
    for link in links:
        nforeLink_pos[link[1]] += 1
        nbackLink_pos[link[2]-1] += 1
    if( index > 0 ):
        for i in range(index):
            nforeLink_entp[i] = entropy(nforeLink_pos[i] / (index - i))
            nbackLink_entp[i] = entropy(nbackLink_pos[i] / (i + 1))
        maxlink = max(max(nbackLink_pos), max(nforeLink_pos))
        current_state.append([nbackLink_pos[index-1]/maxlink, 1-nbackLink_pos[index-1]/maxlink])
    index += 1
    # print(nforeLink_pos)
    # print(nbackLink_pos)
    # print(nforeLink_entp)
    # print(nbackLink_entp)

print(current_state)
# img = np.zeros(400,400,3), np.uint8
# img = cv2.rectangle(img, (10,10), (100,100), (125,125,125), 3)