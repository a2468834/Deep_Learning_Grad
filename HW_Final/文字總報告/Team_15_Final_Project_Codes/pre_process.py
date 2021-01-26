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
import seaborn
from   sklearn.preprocessing import LabelEncoder
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
    subm_col   = lambda : ['ID', '預測乳量']
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
        
        # Use the column 'ID' as the index of DataFrame
        input_df = input_df.set_index('ID')
        
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
        
    elif 'subm' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.subm_col())
        
        # Use the column 'ID' as the index of DataFrame
        input_df = input_df.set_index('ID')
        
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


def splitREPORT(report_df):
    # Extract testing data from raw report.csv
    train_report_df = report_df[report_df['乳量'].notnull()]
    test_report_df = report_df[report_df['乳量'].isnull()]
    
    # Inplace processing
    train_report_df = processREPORT(train_report_df)
    test_report_df = processREPORT(test_report_df)
    
    return train_report_df, test_report_df


def processREPORT(partial_report):
    # Sort training data of report_df by '乳牛編號'
    partial_report = partial_report.sort_values(by=['乳牛編號', '採樣日期'])
    
    # Drop the following columns because they are the duplicates of '採樣日期'
    drop_columns = ['資料年度', '資料月份', '檢測日期']
    #partial_report = partial_report.drop(drop_columns, axis=1)
    
    # Drop the following columns because they are the duplicates of '配種次數'
    drop_columns = ['最後配種日期', '最後配種精液', '第一次配種日期', '第一次配種精液']  
    #partial_report = partial_report.drop(drop_columns, axis=1)
    
    # Add a new empty column to the DataFrame
    partial_report['分娩次數'] = 0
    
    # Convert 最近分娩日期' to '分娩次數'
    curr_cow_id, birth_dates = 0, []
    for index, row in partial_report.iterrows():
        if curr_cow_id == row['乳牛編號']:
            if row['最近分娩日期'] in birth_dates:
                partial_report.loc[index, '分娩次數'] = len(birth_dates)
            else:
                birth_dates.append(row['最近分娩日期'])
                partial_report.loc[index, '分娩次數'] = len(birth_dates)
        else: # Change for calculating another cow_id's '分娩次數'
            curr_cow_id     = row['乳牛編號']
            birth_dates     = []
            birth_dates.append(row['最近分娩日期'])
            partial_report.loc[index, '分娩次數'] = len(birth_dates)
    
    # Drop the following columns because they are the duplicates of '分娩次數'
    drop_columns = ['最近分娩日期', '前次分娩日期']
    #partial_report = partial_report.drop(drop_columns, axis=1)
    
    # Deal with the rest of columns
    partial_report['泌乳天數'].fillna(value=1, inplace=True)
    partial_report['父親牛精液編號'].fillna(value='UNKNOWN', inplace=True)
    partial_report['母親乳牛編號'].fillna(value='UNKNOWN', inplace=True)
    partial_report['泌乳天數'].fillna(value=0, inplace=True)
    partial_report['最後配種精液'].fillna(value='UNKNOWN', inplace=True)
    partial_report['第一次配種精液'].fillna(value='UNKNOWN', inplace=True)
    
    return partial_report


def plotHeatmap(df_corr_matrix, img_name=None):
    xticks = ['C{}'.format(i) for i in range(len(df_corr_matrix.columns))]
    yticks = ['C{}'.format(i) for i in range(len(df_corr_matrix.index))]
    '''
    for i, column in zip(range(len(df_corr_matrix.columns)), df_corr_matrix.columns):
        print("C{} is {}".format(i, column))
    df_corr_matrix.abs().to_csv('corr_matrix.csv', encoding='utf_8_sig')
    '''
    pyplot.pcolor(df_corr_matrix.abs(), cmap='hot')
    pyplot.xticks(numpy.arange(0.5, len(df_corr_matrix.columns), 1), xticks)
    pyplot.yticks(numpy.arange(0.5, len(df_corr_matrix.index), 1), yticks)
    if img_name is None:
        pyplot.show()
    else:
        pyplot.savefig('{}.png'.format(img_name), dpi=300)


########## Main function ##########
if __name__ == '__main__':
    # Display pandas DataFrame without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    
    # Read .csv
    report_df = readCSV('./data/report.csv')
    birth_df  = readCSV('./data/birth.csv')
    breed_df  = readCSV('./data/breed.csv')
    spec_df   = readCSV('./data/spec.csv')
    subm_df   = readCSV('./data/submission.csv')
    
    train_report_df, test_report_df = splitREPORT(report_df)
    
    with open('train_report_df.pk', 'wb') as f:
        pickle.dump(train_report_df, f)
    with open('test_report_df.pk', 'wb') as f:
        pickle.dump(test_report_df, f)
    
    
    pearson_corr_matrix = train_report_df.corr(method ='pearson')
    kendall_corr_matrix = train_report_df.corr(method ='kendall')
    spearman_corr_matrix = train_report_df.corr(method ='spearman')
    
    plotHeatmap(pearson_corr_matrix, img_name='pearson')
    #plotHeatmap(kendall_corr_matrix, img_name='kendall')
    #plotHeatmap(spearman_corr_matrix, img_name='spearman')
    exit()
    
    # Merge all DFs
    combined_dict = {} # 4-tuple of dict
    all_cow_id = set(report_df['乳牛編號']).union(set(birth_df['乳牛編號']), set(breed_df['乳牛編號']), set(spec_df['乳牛編號']))
    
    for cow_id in all_cow_id:
        combined_dict[cow_id] = (report_df.loc[report_df['乳牛編號'] == cow_id], 
                                 birth_df.loc[birth_df['乳牛編號'] == cow_id], 
                                 breed_df.loc[breed_df['乳牛編號'] == cow_id],
                                 spec_df.loc[spec_df['乳牛編號'] == cow_id])
    
    # Drop all 'cow_id' which have empty report_DataFrame
    delete_keys = []
    for key in combined_dict.keys():
        if combined_dict[key][0].empty:
            delete_keys.append(key)
    for key in delete_keys:
        del combined_dict[key]
    
    
    # Low quality codes
    combined_df = report_df
    # Add new columns
    df_length = len(combined_df['乳牛編號'])
    df_index  = combined_df.index
    for col_name in (CONST.birth_col()+CONST.breed_col()+CONST.spec_col()):
        combined_df.loc[:, col_name] = pandas.Series(numpy.zeros(df_length), index=df_index)
    
    for key in combined_dict.keys():
        for birthday in combined_dict[key][1]['分娩日期']:
            combined_df[combined_df['最近分娩日期'] == birthday][CONST.birth_col()] = combined_dict[key][1]
    # Low quality codes
    
    print(combined_df['乳牛編號'==52612])
    
    
    with open('combined_dict.pk', 'wb') as f:
        pickle.dump(combined_dict, f)
