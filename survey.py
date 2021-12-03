import pandas as pd
import math
import numpy as np

test_csv = pd.read_csv(
    'exp_log/1_feedback_viewedFP_caadria_이정빈.csv')
fpname = test_csv['floorplan_name']
eventname = test_csv['event']

test_csv['mode'] ='a'
# test_csv['mode'] = test_csv.astype({'mode':'str'})
test_csv['priority'] = 'a'
sample_mode = test_csv['mode']
sample_priority = test_csv['priority']
index = 0
find_row = test_csv.loc[(test_csv['event']== 1)]
con = 'con'
div = 'div'
no = 'no'
mode_info = [[div,con,con,div,con,con,con],[div,div,con,con,con,con,con],[con,div,con,con,con,div,con],[con,div,con,con,con,div,con],[con,div,con,con,con,div,con],[con,div,con,con,con,div,con],[con,div,div,con,con,con,con],[con,con,div,con,con,div,con],[div,con,con,con,con,div,con],[con,con,div,con,con,div,con]]
priority_info = [[6,2,4,3,5,1],[6,2,5,1,3,4],[6,3,5,2,1,4],[6,3,5,2,1,4],[6,3,5,2,1,4],[6,3,5,2,1,4],[6,5,4,3,1,2],[6,1,4,3,2,5],[6,3,5,1,2,4],[6,1,3,4,2,5]]
mode_index = 0
for i in eventname:
    if i == 1:
        test_csv.at[index,'mode'] = mode_info[mode_index]
        test_csv.at[index,'priority'] = priority_info[mode_index]
        mode_index += 1
    index += 1
print(test_csv['mode'])
test_csv.to_csv('exp_log/1_feedback_viewedFP_caadria_이정빈_survey.csv')


# 이정빈
mode_info = [[div,con,con,div,con,con,con],[div,div,con,con,con,con,con],[con,div,con,con,con,div,con],[con,div,con,con,con,div,con],[con,div,con,con,con,div,con],[con,div,con,con,con,div,con],[con,div,div,con,con,con,con],[con,con,div,con,con,div,con],[div,con,con,con,con,div,con],[con,con,div,con,con,div,con]]
priority_info = [[6,2,4,3,5,1],[6,2,5,1,3,4],[6,3,5,2,1,4],[6,3,5,2,1,4],[6,3,5,2,1,4],[6,3,5,2,1,4],[6,5,4,3,1,2],[6,1,4,3,2,5],[6,3,5,1,2,4],[6,1,3,4,2,5]]
# 박은광
# mode_info = [[div,div,div,div,no,con,div],[con,con,div,con,no,div,con],[con,con,con,div,no,con,con],[div,con,con,div,no,con,div],[div,div,con,div,div,con,div],[no,con,div,div,con,div,div],[div,div,div,div,con,div,div],[div,con,div,div,con,div,con],[div,div,div,div,con,con,div],[div,div,div,div,con,con,con]]
# priority_info = [[1,4,5,3,6,2],[2,5,4,1,6,3],[6,4,1,2,5,3],[5,1,4,2,6,3],[6,4,3,5,2,1],[5,2,3,4,1,6],[6,1,5,4,2,3],[4,1,5,3,2,6],[1,5,6,2,4,3],[6,2,4,5,3,1]]
# 손허원
# mode_info = [[con,no,con,div,div,con,div],[div,div,con,div,div,con,div],[con,div,con,div,div,div,div],[con,div,div,div,div,con,div],[div,con,div,div,no,con,div],[div,div,con,div,div,con,div],[con,div,no,div,div,div,div],[con,con,con,con,div,no,con],[div,con,div,con,div,con,con],[con,con,con,con,con,con,con]]
# priority_info = [[4,6,5,3,2,1],[5,1,6,4,3,2],[5,2,6,4,1,3],[6,4,5,3,1,2],[5,3,4,2,6,1],[6,5,3,4,1,2],[5,3,4,6,2,1],[6,4,5,2,3,1],[3,2,5,6,4,1],[3,6,5,4,2,1]]
# 김예나
# mode_info = [[div,div,div,con,div,con,div],[div,con,div,con,div,con,div],[div,div,con,con,div,con,div],[div,con,div,con,con,con,div],[div,div,div,div,con,con,div],[con,con,div,div,con,con,div],[con,con,con,con,con,con,con],[div,con,con,div,con,con,div],[div,con,con,div,con,con,div],[con,con,con,con,con,con,con]]
# priority_info = [[5,3,4,2,6,1],[5,3,4,2,6,1],[4,5,3,2,6,1],[6,5,4,2,3,1],[6,5,4,3,2,1],[4,3,6,5,2,1],[6,5,4,3,2,1],[6,3,4,5,2,1],[6,3,4,5,2,1],[6,5,4,3,2,1]]
# 이민정
# mode_info = [[div,div,div,div,con,con,div],[div,con,div,con,con,div,con],[div,div,div,con,con,div,con],[div,div,div,con,con,div,con],[div,con,div,con,con,div,div],[div,div,div,div,con,div,div],[div,div,div,con,con,div,div],[div,div,div,con,con,con,con],[con,div,div,con,con,con,con],[div,div,con,con,con,div,con]]
# priority_info = [[6,2,4,1,5,3],[6,2,5,1,3,4],[6,3,4,1,2,6],[4,3,5,1,2,6],[6,3,4,1,2,5],[1,3,6,2,4,5],[6,3,4,1,2,5],[6,5,4,1,2,3],[6,2,5,1,3,4],[4,3,5,1,2,6]]