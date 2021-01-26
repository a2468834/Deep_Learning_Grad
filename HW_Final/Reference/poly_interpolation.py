#  Date:         2020/12/01
#  CourseID:     10910COM 526000
#  Course:       Deep Learning (Graduated)
#
#  Environment:
#    [Configuration 1]
#      SW:  Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      HW:  Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
#    [Configuration 2]
#      SW:  Python 3.8.5 on Ubuntu 20.04.1 LTS (Linux 5.4.0-54-generic x86_64)
#      HW:  AMD Ryzen 5 3400G, 64GB DDR4 non-ECC ram, and no discrete GPU
import datetime
import itertools
import math
import matplotlib.pyplot as pyplot
from   multiprocessing import Pool
import numpy
import os
import pandas
import pickle
import random
import re
from   sklearn.preprocessing import LabelEncoder
from   sklearn.manifold import TSNE
import time
import torch
from   torch import optim
import zlib

########## Classes ##########
class CONST:
    # ID means '鮮奶的ID'
    report_col = lambda : ['ID', '資料年度', '資料月份', '酪農場代號', '乳牛編號', '父親牛精液編號', 
                           '母親乳牛編號', '出生日期', '胎次', '泌乳天數', '乳量', '最近分娩日期', 
                           '採樣日期', '月齡', '檢測日期', '最後配種日期', '最後配種精液', 
                           '配種次數', '前次分娩日期', '第一次配種日期', '第一次配種精液']
    birth_col  = lambda : ['乳牛編號', '分娩日期', '乾乳日期', '犢牛編號1', '犢牛編號2', '母牛體重', 
                           '登錄日期', '計算胎次', '胎次', '分娩難易度', '犢牛體型', '犢牛性別', 
                           '酪農場代號']
    breed_col  = lambda : ['乳牛編號', '配種日期', '配種精液', '登錄日期', '孕檢', '配種方式', 
                           '精液種類', '酪農場代號']
    spec_col   = lambda : ['乳牛編號', '狀況類別', '狀況代號', '狀況日期', '備註', '登錄日期', 
                           '酪農場代號']
    dupl_col   = lambda : ['乳牛編號', '胎次', '登錄日期', '酪農場代號']
    unix_time0 = lambda : datetime.datetime(1970, 1, 1)

'''
[Drop]
report: 資料年度、資料月份、出生日期(被月齡取代)、採樣日期、檢測日期、前次分娩日期(被最近分娩日期取代)
birth: 犢牛編號1/2、登錄日期、計算胎次
breed: 登錄日期、孕檢
spec: 登錄日期、備註

[同隻乳牛相減轉間隔]
report: 最近分娩日期、最後配種日期
birth: 分娩日期、乾乳日期
breed: 配種日期
spec: 狀況日期
'''



########## Functions ##########
def readCSV(file_path):
    fp = open(file_path, 'r', encoding='utf-8')
    
    if 'report' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.report_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['出生日期', '最近分娩日期', '採樣日期', '檢測日期', '最後配種日期', '前次分娩日期', '第一次配種日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    elif 'birth' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.birth_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['分娩日期', '乾乳日期', '登錄日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    elif 'breed' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.breed_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['配種日期', '登錄日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    elif 'spec' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.spec_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['狀況日期', '登錄日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    else:
        print("Try to open an unknown .csv file.")
        input_df = None
        fp.close()
        exit()
    
    fp.close()
    return input_df


def str2UnixEpoch(date_str):
    date_unix = 0
    
    if type(date_str) == str:
        match = re.match(r'(?P<y>\d+)\/(?P<m>\d+)\/(?P<d>\d+)\s+(?P<hour>\d+)\:(?P<minute>\d+)', date_str)
        date_datetime = datetime.datetime(year=int(match.group('y')), 
                                          month=int(match.group('m')), 
                                          day=int(match.group('d')), 
                                          hour=int(match.group('hour')), 
                                          minute=int(match.group('minute')))
        date_unix = int((date_datetime - CONST.unix_time0()).total_seconds())
    else: # date_str == nan
        date_unix = 0
    
    return date_unix


def readCowDataCSV(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        input_df = pandas.read_csv(fp, index_col=[0], low_memory=False)
        input_df.reset_index(drop=True, inplace=True)
        return input_df


########## Main function ##########
if __name__ == '__main__':
    # Display pandas DataFrame without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    
    # Read .csv
    #data_df = readCowDataCSV('./data/cow_data.csv')
    report_df = readCSV('./data/report.csv')
    report_df = report_df.set_index('ID')
    
    report_df = report_df[['乳牛編號', '採樣日期', '乳量']]
    cow_id_set = set(report_df['乳牛編號'])
    
    # Sort
    report_df = report_df.sort_values(by=['乳牛編號', '採樣日期'])
    
    index = 0
    for cow_id in cow_id_set:
        one_cow_df = report_df.loc[report_df['乳牛編號']==cow_id]
        
        if (pandas.isnull(one_cow_df['乳量'])).any():
            x = one_cow_df[one_cow_df['乳量'].notnull()]['採樣日期'].to_numpy()
            y = one_cow_df[one_cow_df['乳量'].notnull()]['乳量'].to_numpy()
            need_pred = one_cow_df[one_cow_df['乳量'].isnull()][['採樣日期', '乳量']]
            
            
            if x.shape[0] < 4:
                z    = numpy.polyfit(x, y, 2)
                poly = numpy.poly1d(z)
            else:
                z    = numpy.polyfit(x, y, 3)
                poly = numpy.poly1d(z)
            
            for index in need_pred.index:
                aa = poly(need_pred.loc[index]['採樣日期'])
                print(aa)
            print()
            print()
            
        if index > 10:
            exit()
        index += 1
    
    
    
    



########## Other codes ##########
'''
combined_dict = {cow_id : (report_DataFrame, birth_DataFrame, breed_DataFrame, spec_DataFrame)}
                           └─ the slice of report DataFrame to specific cow id
'''
'''
sorted_key = list(combined_dict.keys())
sorted_key.sort()
for key, index in zip(sorted_key, range(len(sorted_key))):
    print('乳牛編號: {}\n'.format(key))
    print(combined_dict[key][0])
    print()
    print(combined_dict[key][1])
    print()
    print(combined_dict[key][2])
    print()
    print(combined_dict[key][3])
    print("################################################################################")
'''
'''
# One-Hot encoding
data_df = pandas.get_dummies(data_df, prefix_sep='=', dummy_na=True, drop_first=False)

# DataFrame to Numpy-ndarray
data_npy = data_df.to_numpy()
'''
'''
# Drop all non-numerical columns
drop_columns = []
for col_name in data_df.columns:
    if data_df[col_name].dtype == 'object':
        drop_columns.append(col_name)
data_df = data_df.drop(columns=drop_columns)
print(data_df.info(verbose=True))
exit()

# t-SNE
data_npy_tsne = TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(data_npy)
'''