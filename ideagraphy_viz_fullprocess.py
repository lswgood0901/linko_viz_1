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

# Class Link for calculation.
class Link:
    type = ""
    linkType = ""
    left = -1
    right = -1
    # Define creator function

    def __init__(self, type, linkType, left, right):
        self.type = type
        self.linkType = linkType
        self.left = left
        self.right = right

user_designMove_list = []
numberLinks = []
overallshapeLinks = []
locationLinks = []
connectivityLinks = []
roomshapeLinks = []
roomshapelayoutLinks = []
all_fore_entropy=[]
all_back_entropy=[]


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
def entropy(pos):
    if pos == 0 or pos == 1:
        return 0
    return -1 * pos * math.log2(pos) - 1 * (1 - pos) * math.log2(1 - pos)
# Generates links
def check_number_link(similarity):
    numberMinimumSimilarity = 0.779135
    # Check the similarity is bigger than number CriticalValue
    if (similarity >= numberMinimumSimilarity):
        return True
    else:
        return False
def check_overallshape_link(similarity):
    overallshapeMinimumSimilarity = 0.191026
    # Check the similarity is bigger than overallshape CriticalValue
    if (similarity <= overallshapeMinimumSimilarity):
        return True
    else:
        return False
def check_location_link(similarity):
    locationMinimumSimilarity = 23.22471
    if similarity <= locationMinimumSimilarity:
        return True
    else:
        return False
def check_connectivity_link(similarity):
    connectivityMinimumSimilarity = 8.653005
    if similarity <= connectivityMinimumSimilarity:
        return True
    else:
        return False
def check_roomshape_link(similarity):
    roomshapeMinimumSimilarity = 0.954387  # 임계유사도 교체 필요
    if similarity <= roomshapeMinimumSimilarity:
        return True
    else:
        return False
def check_roomshapelayout_link(similarity):
    roomshapelayoutMinimumSimilarity = 0.612734  # 임계유사도 교체 필요
    if similarity <= roomshapelayoutMinimumSimilarity:
        return True
    else:
        return False
def calculate_full_entropy(Links, designMoveNumber):

    foreLink_pos = [0 for i in range(designMoveNumber - 1)]
    backLink_pos = [0 for i in range(designMoveNumber - 1)]
    # horizontalLink_pos = [0 for i in range(designMoveNumber - 1)]
    # Count foreLink, backLink, horizontalLink number in this linkography
    for item in Links:
        # print(item.type + "  " + str(item.left) + " | " + str(item.right))
        # print("fore_pos : " + str(foreLink_pos) + "   " + "back_pos : " +
        #       str(backLink_pos) + "    " + "horz_pos : " + str(horizontalLink_pos))
        foreLink_pos[item.left] += 1
        backLink_pos[item.right-1] += 1
        # horizontalLink_pos[item.right - item.left - 1] += 1
    # Calculate foreLink, backLink, HorizontalLink possibility in each index
    for i in range(designMoveNumber-1):
        foreLink_pos[i] = foreLink_pos[i] / (designMoveNumber - i - 1)
        backLink_pos[i] = backLink_pos[i] / (i + 1)
        # horizontalLink_pos[i] = horizontalLink_pos[i] / \
        #     (designMoveNumber - i - 1)
        # print(str(i) + " th")
        # print("fore_pos : " + str(foreLink_pos) + "   " + "back_pos : " +
        #       str(backLink_pos) + "    " + "horz_pos : " + str(horizontalLink_pos))

    Total_Fore_Entropy = 0
    Total_Back_Entropy = 0
    # Calculate entropy
    for i in range(designMoveNumber-1):
        Total_Fore_Entropy += entropy(foreLink_pos[i])
        Total_Back_Entropy += entropy(backLink_pos[i])
        # TotalEntropy += entropy(horizontalLink_pos[i])

    return Total_Fore_Entropy, Total_Back_Entropy
def add_and_update_ideagraphy_caadria(reference_name):
    global user_designMove_list

    global numberLinks
    global overallshapeLinks
    global locationLinks
    global connectivityLinks
    global roomshapeLinks
    global roomshapelayoutLinks
    global all_fore_entropy
    global all_back_entropy

    user_designMove_list.append(reference_name)
    # if first viewed references, return
    if len(user_designMove_list) == 1:
        return
    length = len(user_designMove_list)

    origin_dataframe = pd.read_csv(
        'similarity/' + reference_name + '.csv', index_col=0)
    sim_dataframe = origin_dataframe.copy(
    ).loc[origin_dataframe['floorplan_name'].isin([user_designMove_list[0]])]
    for i in range(1, len(user_designMove_list)-1):
        new_sim_dataframe = origin_dataframe.copy(
        ).loc[origin_dataframe['floorplan_name'].isin([user_designMove_list[i]])]
        sim_dataframe = pd.concat([sim_dataframe, new_sim_dataframe])
    # add links
    right = len(user_designMove_list)-1
    node_type = "Reference_Reference"
    for left in range(len(user_designMove_list)-1):
        # generate links with newly added reference design move
        # try:
        if check_number_link(sim_dataframe.iloc[left]["nr_sim"]):
            numberLinks.append(Link("number", node_type, left, right))
        if check_overallshape_link(sim_dataframe.iloc[left]["fs_sim"]):
            overallshapeLinks.append(
                Link("overallshape", node_type, left, right))
        if check_location_link(sim_dataframe.iloc[left]["rl_sim"]):
            locationLinks.append(Link("location", node_type, left, right))
        if check_connectivity_link(sim_dataframe.iloc[left]["rc_sim"]):
            connectivityLinks.append(
                Link("connectivity", node_type, left, right))
        if check_roomshape_link(sim_dataframe.iloc[left]["rs_sim"]):
            roomshapeLinks.append(Link("roomshape", node_type, left, right))
        if check_roomshapelayout_link(sim_dataframe.iloc[left]["rsl_sim"]):
            roomshapelayoutLinks.append(
                Link("roomshapelayout", node_type, left, right))
    nr_fore_entropy, nr_back_entropy = calculate_full_entropy(numberLinks, length)
    fs_fore_entropy, fs_back_entropy = calculate_full_entropy(overallshapeLinks, length)
    rl_fore_entropy, rl_back_entropy = calculate_full_entropy(locationLinks, length)
    rc_fore_entropy, rc_back_entropy = calculate_full_entropy(connectivityLinks, length)
    rs_fore_entropy, rs_back_entropy = calculate_full_entropy(roomshapeLinks, length)
    rsl_fore_entropy, rsl_back_entropy = calculate_full_entropy(roomshapelayoutLinks, length)
    all_fore_entropy = [nr_fore_entropy, fs_fore_entropy, rl_fore_entropy, rc_fore_entropy, rs_fore_entropy, rsl_fore_entropy]
    all_back_entropy = [nr_back_entropy, fs_back_entropy, rl_back_entropy, rc_back_entropy, rs_back_entropy, rsl_back_entropy]
    return
def each_links_number():
    global user_designMove_list
    global numberLinks
    global overallshapeLinks
    global locationLinks
    global connectivityLinks
    global roomshapeLinks
    global roomshapelayoutLinks

    global link_all
    link_all = []
    # designMoveNumber = len(user_designMove_list)
    # foreLink_pos = [0 for i in range(designMoveNumber - 1)]
    # foreLink_entp = [0 for i in range(designMoveNumber - 1)]
    # backLink_pos = [0 for i in range(designMoveNumber - 1)]
    # backLink_entp = [0 for i in range(designMoveNumber - 1)]
    # Count foreLink, backLink, entropy in this linkography
    all_links = [numberLinks, overallshapeLinks, locationLinks,
                 connectivityLinks, roomshapeLinks, roomshapelayoutLinks]
    for links in all_links:
        for link in links:
            if link.type == "location":
                linktype = "lc"
            elif link.type == "overallshape":
                linktype = "fs"
            elif link.type == "connectivity":
                linktype = "rc"
            elif link.type == "roomshapelayout":
                linktype = "rsl"
            elif link.type == "number":
                linktype = "rn"
            elif link.type == "roomshape":
                linktype = "rs"
            # foreLink_pos[item.left] += 1
            # backLink_pos[item.right-1] += 1
            link_all.append([linktype, link.left, link.right])
    # Count foreLink, backLink, entropy in this linkography
    # for i in range(designMoveNumber-1):
    #     foreLink_entp[i] = entropy(
    #         foreLink_pos[i] / (designMoveNumber - i - 1))
    #     backLink_entp[i] = entropy(
    #         backLink_pos[i] / (i + 1))
    return
def count_linknumber(anylinks, data_length):
    index = 0
    sum_fore = 0
    sum_back = 0
    nforeLink_pos = []
    nbackLink_pos = []
    nforeLink_pos = [0 for i in range(data_length)]
    nbackLink_pos = [0 for i in range(data_length)]
    if len(anylinks) > 0:
        for link in anylinks:
            nforeLink_pos[link.left] += 1
            nbackLink_pos[link.right] += 1
        index += 1
        sum_fore = sum(nforeLink_pos)
        sum_back = sum(nbackLink_pos)
    # nforeLink_pos.append(0)
    # nbackLink_pos.insert(0,0)
    return nforeLink_pos, nbackLink_pos
def appending_currentstate(anylinks):
    index = 0
    fore_ratio = 0
    back_ratio = 0
    current_state = []
    link_length = len(anylinks)
    for links in anylinks:
        nforeLink_pos = [0 for i in range(index)]
        nbackLink_pos = [0 for i in range(index)]
        for link in links:
            nforeLink_pos[link[0]] += 1
            nbackLink_pos[link[1]-1] += 1
        if( index == 0):
            current_state.append([nforeLink_pos[index]/(link_length-1), 0])
        elif( index == link_length):
            current_state.append([0,nbackLink_pos[index]/index-1])
        else:
            current_state.append([nforeLink_pos[index]/(link_length-1), nbackLink_pos[index]/index-1])
        index += 1
        x = 0
        for i in current_state:
            fore_ratio = fore_ratio+i[x][0]
            back_ratio = back_ratio+i[x][1]
            x += 1
    return fore_ratio, back_ratio

def mode_to_number(x):
    y = "wrong"
    if x == "'con'" or x == "con" or x =="' con'" or x =="  'con'" or x == " 'con'":
        y = 1
    elif x == "' div'" or x =="'div'" or x =="div" or x =="  'div'" or x == " 'div'":
        y = -1
    elif x == "' no'" or x == "'no'" or x == "no" or x =="  'no'" or x == " 'no'":
        y = 0
    else:
        print("wrong text:",x)
    return y

# seungwon test
test_csv = pd.read_csv(
    'exp_log/1_feedback_viewedFP_caadria_이채민_survey.csv')
fpname = test_csv['floorplan_name']
eventname = test_csv['event']
user_mode_df = test_csv['mode']
user_mode_df = user_mode_df.values.tolist()
user_priority_df = test_csv['priority']
user_priority_df = user_priority_df.values.tolist()

user_mode_sample = []
user_mode = []
user_priority = []
user_priority_sample = []
index = 0
for i in user_mode_df:
    if eventname[index] == 1:
        user_mode_sample.append(i)
        user_priority_sample.append(user_priority_df[index])
    index += 1
for i in user_mode_sample:
    x = i[1:-1].split(',')
    user_mode.append(x)
for i in user_priority_sample:
    x = i[1:-1].split(',')
    user_priority.append(x)

rn_mode=[]
fs_mode=[]
rl_mode=[]
rc_mode=[]
rs_mode=[]
rsl_mode=[]
rn_priority = []
fs_priority = []
rl_priority = []
rc_priority = []
rs_priority = []
rsl_priority = []
for i in user_priority:
    rn_priority.append(i[5])
    fs_priority.append(i[0])
    rl_priority.append(i[1])
    rc_priority.append(i[4])
    rs_priority.append(i[2])
    rsl_priority.append(i[3])
index = 0
for i in user_mode:
    x = mode_to_number(i[5])
    rn_mode.append(x)
    x = mode_to_number(i[0])
    fs_mode.append(x)
    x = mode_to_number(i[1])
    rl_mode.append(x)
    x = mode_to_number(i[4])
    rc_mode.append(x)
    x = mode_to_number(i[2])
    rs_mode.append(x)
    x = mode_to_number(i[3])
    rsl_mode.append(x)
    index += 1
index = 0
all_mode = [rn_mode, fs_mode, rl_mode, rc_mode, rs_mode, rsl_mode]
all_priority = [rn_priority, fs_priority, rl_priority, rc_priority, rs_priority, rsl_priority]
select_fp = []
temp_fp = [0]
temp2_fp = []
test_index = 0
reset_index = 0
for i in fpname:
    if eventname[test_index] == 1:
        temp_fp.append(test_index)
    temp2_fp.append(i)
    test_index += 1

for i in range(10):
    temp_array = []
    for l in range(temp_fp[i],temp_fp[i+1]+1):
        temp_array.append(temp2_fp[l])
    select_fp.append(temp_array)

# 링크 다시 생성
rnlinks = []
fslinks = []
lclinks = []
rslinks = []
rsllinks = []
rclinks = []
index = 0
linkdata = []
selected_data = []
for i in fpname:
    add_and_update_ideagraphy_caadria(i)
    each_links_number()
    index += 1
index = 0
for a in range(len(fpname)):
    rnlink = []
    fslink = []
    lclink = []
    rslink = []
    rsllink = []
    rclink = []
    for b in link_all:
        if b[0] == "rn":
            rnlink.append([b[1], b[2]])
        elif b[0] == "fs":
            fslink.append([b[1], b[2]])
        elif b[0] == "lc":
            lclink.append([b[1], b[2]])
        elif b[0] == "rs":
            rslink.append([b[1], b[2]])
        elif b[0] == "rsl":
            rsllink.append([b[1], b[2]])
        elif b[0] == "rc":
            rclink.append([b[1], b[2]])
        else:
            print("wrong keyword")
    selected_data.append(eventname[a])
    rnlinks.append(rnlink)
    fslinks.append(fslink)
    lclinks.append(lclink)
    rslinks.append(rslink)
    rsllinks.append(rsllink)
    rclinks.append(rclink)
    index += 1

select_index = []
for i in range(len(fpname)):
    if eventname[i] == 1:
        select_index.append(i)
all_linko = [rnlinks[0], fslinks[0], lclinks[0], rclinks[0], rslinks[0], rsllinks[0]]
index=0

print(rclinks[0])

x_axis = range(len(select_fp))
name = ['NR_Ideagraphy', 'FS_Ideagraphy', 'RL_Ideagraphy', 'RC_Ideagraphy', 'RS_Ideagraphy', 'RSL_Ideagraphy', 'Mean_Ideagraphy']
plt.rc("font", size=8)
plt.rc('ytick', labelsize=5)
plt.rc('xtick', labelsize=5)
plt.rcParams["figure.figsize"] = (14, 7.5)
# plt.rcParams['axes.grid'] = True
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.55)

index=0
for linko in range(6):
    for links in all_linko[linko]:
        x = (links[0] + links[1])/2
        y = links[1] - links[0]
        plt.plot([links[0], x], [0,y], color="grey", linewidth='0.5')
        plt.plot([links[1],x],[0,y], color="grey", linewidth='0.5')
        plt.scatter(x, y, s=20, c='g')
    index = 0
    for i in range(len(select_index)):
        if i != len(select_index)-1:
            x = (select_index[i] + select_index[i + 1]) / 2
            y = select_index[i + 1] - select_index[i]
            plt.plot([select_index[i], x], [0, y], color="black", linewidth='1')
            plt.plot([select_index[i + 1], x], [0, y], color="black", linewidth='1')
        if i == 0:
            x = (select_index[i]) / 2
            y = select_index[i]
            plt.plot([0, x], [0, y], color="black", linewidth='1')
            plt.plot([select_index[i], x], [0, y], color="black", linewidth='1')
    index = 0
    for i in range(len(fpname)):
        plt.scatter(i, 0, s=20, c='black')
        if eventname[i] == 1:
            if all_mode[linko][index] == 1:
                plt.scatter(i, -1, s=20, c='r')
            elif all_mode[linko][index] == -1:
                plt.scatter(i, -1, s=20, c='b')
            elif all_mode[linko][index] == 0:
                plt.scatter(i, -1, s=20, c='black')
            index += 1
    plt.title(name[linko])
    plt.show()
    # plt.savefig('Figure_condiv/유미선/유미선_'+name[linko]+'_1.png')


