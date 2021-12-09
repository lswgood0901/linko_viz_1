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
    'log/div_result_1.csv')

density = test_csv['forelinks_number(density)']
ideagraphy_type = test_csv['ideagraphy_type']
criteria = test_csv['criteria']
priority = test_csv['priority']

macro = []
micro = []
index = 0
for i in priority:
    if i == 1 and criteria[index] == 1:
        micro.append(density[index])
    elif i == 1 and criteria[index] == 0:
        macro.append(density[index])
    index += 1
index = 0
std_macro = np.std(macro)
std_micro = np.std(micro)







# 전체평균
test_stat, p_val = scipy.stats.kstest(macro, 'norm', args=(np.mean(macro), np.var(macro)**0.5))
test_stat1, p_val1 = scipy.stats.kstest(micro, 'norm', args=(np.var(micro), np.var(micro)**0.5))
print("p_val:",p_val, "p_val1:",p_val1)
print(scipy.stats.ttest_ind(macro, micro, equal_var=False))

