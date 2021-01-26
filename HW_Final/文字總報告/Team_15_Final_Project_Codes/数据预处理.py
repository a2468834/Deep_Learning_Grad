import pandas as pd
import pandas
import numpy as np
import numpy
import re
import datetime
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


########## Functions ##########
# 读取csv文件，日期这些字段，都是需要映射成时间戳的
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
    else:
#         print(date_str)
        date_unix = 0
    
    return date_unix



# Display pandas DataFrame without truncation
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

# Read .csv
report_df = readCSV('./data/report.csv')
birth_df  = readCSV('./data/birth.csv')
breed_df  = readCSV('./data/breed.csv')
spec_df   = readCSV('./data/spec.csv')

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

data1 = pd.read_csv("data/report.csv")
for i in combined_dict.copy().keys():
    if i not in list(data1["5"]):
        combined_dict.pop(i)
print(len(combined_dict))

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def concat(cow_id):
    ex = combined_dict[cow_id]
    ex_report = ex[0]
    ex_birth = ex[1][["分娩日期", "乾乳日期", "分娩難易度", "母牛體重"]] #以分娩日期為base
    ex_breed = ex[2][["配種日期", "配種方式", "精液種類"]] #以配種日期為base(=最後配種日期)
    ex_spec = ex[3][["乳牛編號", "狀況類別", "狀況日期"]] #以乳牛編號為base
    index = ex_report.index
    if ex_birth.values.size != 0:
        for i in range(len(list(ex_report["最近分娩日期"].values))):
            if i == 0:
                id = ex_birth[ex_birth["分娩日期"] == ex_report["最近分娩日期"].values[i]][["乾乳日期", "分娩難易度", "母牛體重"]]
                if id.values.size == 0:
                    id = pd.DataFrame(ex_birth.iloc[:1, 1:].values, columns=["乾乳日期", "分娩難易度", "母牛體重"])
            else:
                id_1 = ex_birth[ex_birth["分娩日期"] == ex_report["最近分娩日期"].values[i]][["乾乳日期", "分娩難易度", "母牛體重"]]
                if id_1.values.size == 0:
                    id_1 = pd.DataFrame(ex_birth.iloc[-1:, 1:].values, columns=["乾乳日期", "分娩難易度", "母牛體重"])
                id = pd.concat([id, id_1], axis=0)
        id.index = index
        new_report = pd.concat([ex_report, id], axis=1)
    else:
        new_report = ex_report

    ex_breed = ex_breed.append(pd.Series({"配種日期": int(0), "配種方式": int(-1), "精液種類": int(-1)}), ignore_index=True)
    for i in range(len(list(ex_report["最後配種日期"].values))):
        if i == 0:
            n_id = ex_breed[ex_breed["配種日期"] == ex_report["最後配種日期"].values[i]][["配種方式", "精液種類"]]
        else:
            n_id_1 = ex_breed[ex_breed["配種日期"] == ex_report["最後配種日期"].values[i]][["配種方式", "精液種類"]]
            n_id = pd.concat([n_id, n_id_1], axis=0)
    n_id.index = index
    new_report = pd.concat([new_report, n_id], axis=1)

    for i in range(len(list(ex_report["乳牛編號"].values))):
        if i == 0:
            m_id = pd.DataFrame(ex_spec[ex_spec["乳牛編號"] == ex_report["乳牛編號"].values[i]][["狀況類別", "狀況日期"]].values.reshape(1, -1))

        else:
            m_id_1 = pd.DataFrame(ex_spec[ex_spec["乳牛編號"] == ex_report["乳牛編號"].values[i]][["狀況類別", "狀況日期"]].values.reshape(1, -1))
            m_id = pd.concat([m_id, m_id_1], axis=0)
    m_id.index = index
    new_report = pd.concat([new_report, m_id], axis=1)
    return new_report

from tqdm import tqdm
cow_id_list = unique(data1.loc[:, "5"].values)
for i in tqdm(range(len(cow_id_list))):
    if i == 0:
        cow_data = concat(cow_id_list[i])
    else:
        cow_data = pd.concat([cow_data, concat(cow_id_list[i])], axis=0)
#         print(i)

df_org = cow_data.copy()
for i, j in zip(range(0, 20, 2), range(1, 20, 2)):
    c_class = "狀況類別" + str(int(i / 2 + 1))
    c_date = "狀況日期" + str(int(i / 2 + 1))
    df_org.rename(columns={i: c_class, j: c_date}, inplace=True)


df = df_org.copy().drop(["資料年度"],axis=1)
df = df.sort_values(by=['ID'])

df["母牛體重"].fillna(df["母牛體重"].mode()[0], inplace=True)
df["分娩難易度"].fillna(df["分娩難易度"].mode()[0], inplace=True)
# df["狀況類別1"].fillna(df["狀況類別1"].mode()[0], inplace=True)
df["出生年份"] = df["出生日期"].apply(lambda x: time.localtime(x).tm_year)
df["出生月份"] = df["出生日期"].apply(lambda x: time.localtime(x).tm_mon)
df["父親牛精液編號"] = pd.factorize(df["父親牛精液編號"])[0].astype(int)
df["母親乳牛編號"] = pd.factorize(df["母親乳牛編號"])[0].astype(int)
df["最後配種精液"] = pd.factorize(df["最後配種精液"])[0].astype(int)

df["精液種類"] = pd.factorize(df["精液種類"])[0].astype(int)
df["配種方式"] = pd.factorize(df["配種方式"])[0].astype(int)
df["酪農場代號"] = pd.factorize(df["酪農場代號"])[0].astype(int)
df["狀況類別1"] = pd.factorize(df["狀況類別1"])[0].astype(int)
df["狀況類別2"] = pd.factorize(df["狀況類別2"])[0].astype(int)
df["狀況類別3"] = pd.factorize(df["狀況類別3"])[0].astype(int)
df["狀況類別4"] = pd.factorize(df["狀況類別4"])[0].astype(int)
df["狀況類別5"] = pd.factorize(df["狀況類別5"])[0].astype(int)
df["狀況類別6"] = pd.factorize(df["狀況類別6"])[0].astype(int)
df["狀況類別7"] = pd.factorize(df["狀況類別7"])[0].astype(int)
df["狀況類別8"] = pd.factorize(df["狀況類別8"])[0].astype(int)
df["狀況類別9"] = pd.factorize(df["狀況類別9"])[0].astype(int)
df["狀況類別10"] = pd.factorize(df["狀況類別10"])[0].astype(int)
# df["乳牛編號"] = pd.factorize(df["乳牛編號"])[0].astype(int)
df["第一次配種精液"] = pd.factorize(df["第一次配種精液"])[0].astype(int)

lookup = {
    11: '0',
    12: '0',
    1: '0',
    2: '1',
    3: '1',
    4: '1',
    5: '2',
    6: '2',
    7: '2',
    8: '3',
    9: '3',
    10: '3'
}
# # 获取季节


df["資料季节"] = df['資料月份'].map(lookup).astype(int)
df["出生季节"] = df['出生月份'].map(lookup).astype(int)
df["分娩间隔"] = (df['最近分娩日期'] - df["前次分娩日期"])
df["出生间隔"] = (df['最近分娩日期'] - df["最後配種日期"])
df["分娩到乾乳"] = (df['最近分娩日期'] - df["乾乳日期"])
df["配種间隔"] = (df['最後配種日期'] - df["第一次配種日期"])

df["泌乳天數*酪農場代號"] = (df['泌乳天數'] * df["酪農場代號"])
df["泌乳天數+酪農場代號"] = (df['泌乳天數'] + df["酪農場代號"])
df["泌乳天數*配種次數"] = (df['泌乳天數'] * df["配種次數"])


time_col = [col for col in df.columns if col not in  ["乳量","ID","乳牛編號"]]
# 数据都使用一下最大最小归一化
time_data  = df[time_col]
min_data = time_data.min()
max_data = time_data.max()
df[time_col] = (time_data - min_data) / (max_data - min_data)
df = df.fillna(-1)
df.to_csv("./test_data4.csv",index=False)