import cv2
import csv
import numpy as np
import pandas as pd
import math
import copy
import json
import ast
import os
import shutil
from matplotlib import pyplot as plt
import time
import bisect
import scipy.stats

test_csv = pd.read_csv(
    'log/div_result.csv')

density = test_csv['forelinks_number(density)']
ideagraphy_type = test_csv['ideagraphy_type']
criteria = test_csv['criteria']
priority = test_csv['priority']

macro = []
macro_two = []
micro = []
micro_two = []
index = 0
for i in priority:
    if i == 1 and criteria[index] == 1:
        macro.append(density[index])
    elif i == 1 and criteria[index] == 0:
        micro.append(density[index])
    index += 1
index = 0
for i in priority:
    if i == 2 and criteria[index] == 1:
        macro_two.append(density[index])
    elif i == 2 and criteria[index] == 0:
        micro_two.append(density[index])
    index += 1
print(len(macro), len(micro))
print(np.mean(macro),np.mean(micro))



