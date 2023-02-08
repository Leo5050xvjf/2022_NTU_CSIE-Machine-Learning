


import pandas as pd
import os
import sys
import numpy as np
import math
from sklearn import ensemble, preprocessing, metrics,model_selection
import random
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import KFold

'''
1.test_ID +train_ID共有7043筆資料
2.test_ID:1409筆資料、train_ID:5634筆資料
3.demographics.csv 、location.csv、satisfaction.csv、services.csv都是6163筆資料
4.status.csv是紀錄續留與否，也就是所謂的labels
5."./MyData.csv"是3.的所有csv concate
6.從MyData.csv抽出需要的特徵並fillna
7.輸出train_data_ver_1.csv，作為接下來RF的input()


'''
# 確認df的shape
def check_csv_file_status():
    csv_path = []
    for path in os.listdir("./"):
        if path.split(".")[-1] == "csv":
            csv_path.append(os.path.join("./",path))


    for p in csv_path:
        filename = p.split("/")[-1]
        df = pd.read_csv(p)

        # if df.shape[0] == 6163:
        #     print(f"{filename} shape is {df.shape}")
        print(f"{filename} shape is {df.shape}")
# check_csv_file_status()

# 確認train的資料，較完整的有多少筆，並回傳完整的資料的id
def check_ID_data_Integrity(train_ID_path,Intrest_feature):
    csv_path = []
    for path in os.listdir("./"):
        if path.split(".")[-1] == "csv":
            csv_path.append(os.path.join("./",path))
    train_ID_df = pd.read_csv(train_ID_path)
    train_ID_list = train_ID_df["Customer ID"].tolist()

    compara_list = []
    for p in csv_path:
        filename = p.split("/")[-1]
        if filename in Intrest_feature:
            compara_list.append(p)
            df = pd.read_csv(p)
            df_list = df["Customer ID"].tolist()
            counter = 0
            for ID in df_list:
                if ID in train_ID_list:
                    counter+=1
            # print(f"{filename} 共有 {len(df_list)}: match :{counter}")
    total_list = []
    for p in compara_list:
        df = pd.read_csv(p)
        df_list = df["Customer ID"].tolist()
        total_list.append(df_list)
    counter = 0

    best_id = []
    bad_id = []

    list_num = len(total_list)
    for t in train_ID_list:
        counter = 0
        for list_ in total_list:
            if t in list_:
                counter+=1
        if counter == list_num:
            best_id.append(t)
            counter+=1
        else:bad_id.append(t)
    # print(f"同時存在的共{counter}筆")
    return best_id,bad_id


# 將回傳的id，從所有找的到的資料夾中取出
def interge_DF(best_id,vs_csv):
    all_features = []
    counter = 0
    for ID in best_id:
        print(counter)
        counter+=1
        feature = [ID]
        for csv in vs_csv:
            df = pd.read_csv(f"./{csv}")
            idx = df.index
            condition =df["Customer ID"] == ID
            id = idx[condition].tolist()[0]
            row = df.loc[id].tolist()
            feature+=row[1:]
        all_features.append(feature)

    all_columns = ["Customer ID"]
    # print("here")
    for csv in vs_csv:
        df = pd.read_csv(f"./{csv}")
        col = df.columns.tolist()
        all_columns+=col[1:]
    df = pd.DataFrame(all_features,columns=all_columns)
    df.to_csv("./MyData.csv",index=False)
# 小工具，確認某df的特徵是不是有nan
def check_isnull(df,feature):
    ans = df[feature].isnull().sum()
    # print(f"features {feature} have {ans} null")
    if ans>0:
        print(f"features {feature} have {ans} null")

# 將決定要train的資料進行整理,把所有缺失項補齊



# 沒啥重要的def
def split_train_test(total_csv,train_id_csv,test_id_csv):
    ori =pd.read_csv(total_csv)
    ori_id = ori["Customer ID"].tolist()
    train_df = pd.read_csv(train_id_csv)
    test_df = pd.read_csv(test_id_csv)
    tra_df = train_df["Customer ID"]
    test_id = test_df["Customer ID"]
# feature = ["demographics.csv", "location.csv", "satisfaction.csv", "services.csv"]
# best_id, bad_id = check_ID_data_Integrity("./Test_IDs.csv", feature)
# all_test_id = best_id+bad_id

def fill_test_id_in_4csv(test_id,four_csv,root = './'):
    test_id_df = pd.DataFrame()
    # record_cll_col = []
    for id in test_id:
        temp_df = pd.DataFrame()
        for csv_file in four_csv:
            csv_path = os.path.join(root,csv_file)
            df = pd.read_csv(csv_path)
            # col = df.columns.tolist()
            # record_cll_col+=col
            c = df['Customer ID'] == id

            id_row = df.loc[c]
            temp_df = temp_df.append(id_row,ignore_index=True)
        # temp_df = pd.concat(temp_df,axis=1)
        # print(temp_df.shape)

        features = [np.nan for _ in range(temp_df.shape[1])]
        column = temp_df.columns.tolist()

        for i,row in temp_df.iterrows():
            val = row.values.tolist()
            for n,v in enumerate(val):
                if pd.notnull(v):
                    features[n] = v

        id_row = pd.DataFrame([features],columns=column)
        test_id_df = test_id_df.append(id_row,ignore_index=True)
    test_id_df.to_csv("./preprocess_test_ID.csv",index=False)
# fill_test_id_in_4csv(all_test_id,feature)



def DATA_Preprocess(train_data_path,test_data_path):
    intresting_features = ["Customer ID","Age","Under 30","Senior Citizen","Lat Long","Latitude","Longitude","Satisfaction Score",
                           "Referred a Friend","Number of Referrals","Tenure in Months","Phone Service",
                           "Multiple Lines","Internet Service","Internet Type","Online Security","Online Backup",
                           "Device Protection Plan","Premium Tech Support","Unlimited Data","Contract","Total Charges",
                           "Total Refunds","Total Extra Data Charges","Total Long Distance Charges","Total Revenue","Churn Category"]
    # 去掉
    # Total Charges	Total Refunds	Total Extra Data Charges	Total Long Distance Charges	Total Revenue
    df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    print(df.shape)
    print(test_df.shape)
    combine_df = pd.concat([df,test_df],ignore_index=True)

    def preprocess_data(df,intresting_features):
        # df.to_csv("./train_ver_1.csv")
        # sys.exit()
        # print(df.shape)
        # print(df.isnull().sum())
        # age_median = [24,49,76]
        # (49+76)/2 = 63
        # print(df.loc[df['Age'] < 30,'Age'].median())
        # print(df.loc[(df['Age'] > 30) &(df['Age'] < 70), 'Age'].median())
        # print(df.loc[df['Age'] > 70, 'Age'].median())


        df.loc[(df['Under 30'] == "Yes")&(df['Age'].isnull()), 'Age'] = 24
        df.loc[(df['Age'] <= 30) & (df['Under 30'].isnull()), 'Under 30'] ="Yes"
        df.loc[(df['Age'] > 30)& (df['Under 30'].isnull()), 'Under 30'] = "No"
        df.loc[(df['Age'].isnull()),'Age'] = 63

        # 以上把年齡做填值

        df.loc[(df['Age'] >= 65)&(df['Senior Citizen'].isnull()), 'Senior Citizen'] = "Yes"
        df.loc[(df['Age'] < 65) & (df['Senior Citizen'].isnull()), 'Senior Citizen'] = "No"
        # check_isnull(df, 'Senior Citizen')
        # 以上把是否是'Senior Citizen'填值

        # ans = df["Latitude"].astype(str)+','+df["Longitude"].astype(str)
        # df.loc[(-df["Lat Long"].isnull())&((df["Latitude"].isnull()) | (df["Longitude"].isnull())),"Latitude"] = []
        df['Latitude'] = df['Latitude'].astype(np.float64)
        df['Longitude'] = df['Longitude'].astype(np.float64)
        # check_isnull(df,'Lat Long')
        # check_isnull(df,'Latitude')
        # check_isnull(df,'Longitude')
        for i in range(df.shape[0]):
            # 左空右皆不空，用右填左
            if (pd.isnull(df.loc[i,'Lat Long']))and (pd.notnull(df.loc[i,'Latitude']) and pd.notnull(df.loc[i,'Longitude'])):
                try:
                    df.loc[i,'Lat Long'] = df.loc[i,'Latitude'].astype(str)+',' +df.loc[i,'Longitude'].astype(str)
                except:
                    df.loc[i, 'Lat Long'] = str(df.loc[i, 'Latitude']) + ',' + str(df.loc[i, 'Longitude'])

                # print(df.loc[i,'Lat Long'])
                # input()
            # 左不空,右任一空
            elif (pd.notnull(df.loc[i,'Lat Long']))and (pd.isnull(df.loc[i,'Latitude']) or pd.isnull(df.loc[i,'Longitude'])):
                l,r = df.loc[i,'Lat Long'].split(',')
                df.loc[i, 'Latitude'] = float(l)
                df.loc[i, 'Longitude'] = float(r)
        # check_isnull(df,'Lat Long')
        # check_isnull(df,'Latitude')
        # check_isnull(df,'Longitude')
        # 36.15254796653061
        # -119.72580400690777
        # a = df['Latitude'].mean()
        # b = df['Longitude'].mean()
        df['Latitude'] = df['Latitude'].fillna(36.15254796653061)
        df['Longitude'] = df['Longitude'].fillna(-119.72580400690777)
        # check_isnull(df,'Lat Long')
        # check_isnull(df,'Latitude')
        # check_isnull(df,'Longitude')

        for i in range(df.shape[0]):
            # 左空右皆不空，用右填左
            if (pd.isnull(df.loc[i,'Lat Long']))and (pd.notnull(df.loc[i,'Latitude']) and pd.notnull(df.loc[i,'Longitude'])):
                try:
                    df.loc[i,'Lat Long'] = df.loc[i,'Latitude'].astype(str)+',' +df.loc[i,'Longitude'].astype(str)
                except:
                    df.loc[i, 'Lat Long'] = str(df.loc[i, 'Latitude']) + ',' + str(df.loc[i, 'Longitude'])

        # check_isnull(df,'Lat Long')
        # check_isnull(df,'Latitude')
        # check_isnull(df,'Longitude')



        #Satisfaction Score

        df.loc[(df['Satisfaction Score'].isnull()) & (df['Churn Category'] == "No Churn") , 'Satisfaction Score'] = round(df['Satisfaction Score'].mean())
        df.loc[(df['Satisfaction Score'].isnull()) & (df['Churn Category'] != "No Churn"), 'Satisfaction Score'] = "0"
        # check_isnull(df,'Satisfaction Score')
        # 以上完成滿意度設置

        df.loc[(df["Referred a Friend"].isnull()) & (df['Satisfaction Score'].astype(int) >= 3),"Referred a Friend"] ="Yes"
        df.loc[(df["Referred a Friend"].isnull()) & (df['Satisfaction Score'].astype(int)  <3),"Referred a Friend"] ="No"

        # check_isnull(df,"Referred a Friend")
        # 以上完成Referred a Friend設置

        # Referred_a_Friend_mean = df["Number of Referrals"].mean() = 1.7952468007312614
        df.loc[(df["Number of Referrals"].isnull()) & (df["Referred a Friend"]=="No"),"Number of Referrals" ] = "0"
        df.loc[(df["Number of Referrals"].isnull()) & (df["Referred a Friend"]=="Yes"),"Number of Referrals" ] = "2"

        # check_isnull(df, "Number of Referrals")

        # 以上完成Number of Referral設置

        df.loc[(df['Phone Service'] == "No") &df["Tenure in Months"].isnull() ,"Tenure in Months"] = "1"
        df.loc[(df['Phone Service'] == "Yes") & df["Tenure in Months"].isnull(), "Tenure in Months"] = df["Tenure in Months"].median()
        df['Phone Service'] = df['Phone Service'].fillna("Yes")
        df["Tenure in Months"] = df["Tenure in Months"].fillna(df["Tenure in Months"].median())
        # check_isnull(df,'Phone Service')
        # check_isnull(df,"Tenure in Months")

        # 以上完成'Phone Service' 、"Tenure in Months" 的設置


        df["Multiple Lines"] = df["Multiple Lines"].fillna(method = 'bfill')
        df["Multiple Lines"] = df["Multiple Lines"].fillna(method = 'ffill')
        # check_isnull(df, "Multiple Lines")

        df.loc[df['Internet Type'] == "None" ,'Internet Service'] ="No"
        df.loc[df['Internet Type'] != "None" ,'Internet Service'] ="Yes"

        df['Internet Service'] = df['Internet Service'].fillna(method = 'ffill')
        df['Internet Service'] = df['Internet Service'].fillna(method = 'bfill')
        df['Internet Type'] = df['Internet Type'].fillna(method = 'ffill')
        df['Internet Type'] = df['Internet Type'].fillna(method = 'bfill')
        # check_isnull(df, 'Internet Service')
        # check_isnull(df, 'Internet Type')

        df['Online Security'] = df['Online Security'].fillna(method = 'ffill')
        df['Online Security'] = df['Online Security'].fillna(method ='bfill')
        df['Online Backup'] = df['Online Backup'].fillna(method ='ffill')
        df['Online Backup'] = df['Online Backup'].fillna(method ='bfill')
        df['Device Protection Plan'] = df['Device Protection Plan'].fillna(method ='ffill')
        df['Device Protection Plan'] = df['Device Protection Plan'].fillna(method ='bfill')
        df['Premium Tech Support'] = df['Premium Tech Support'].fillna(method ='ffill')
        df['Premium Tech Support'] = df['Premium Tech Support'].fillna(method ='bfill')

        df['Unlimited Data'] = df['Unlimited Data'].fillna(method ='ffill')
        df['Unlimited Data'] = df['Unlimited Data'].fillna(method ='bfill')


        quarter_0_25= df['Total Charges'].astype(float).quantile(0.25)
        quarter_0_75 = df['Total Charges'].astype(float).quantile(0.75)
        # print(df['Total Charges'].notnull())
        # input()
        df.loc[((df['Contract'].isnull()) & (df['Total Charges'].notnull() & df['Total Charges'].astype(float) >quarter_0_75)),'Contract'] = "Two Year"
        df.loc[((df['Contract'].isnull()) & (df['Total Charges'].notnull() & (df['Total Charges'].astype(float) < quarter_0_75) &(df['Total Charges'].astype(float)) > quarter_0_25)),'Contract'] = "One Year"
        df.loc[((df['Contract'].isnull()) & (df['Total Charges'].notnull() & df['Total Charges'].astype(float) < quarter_0_25)),'Contract'] = "Month-to-Month"

        df.loc[(df['Contract'] == 'One Year') &(df['Total Charges'].isnull()),'Total Charges'] = str((quarter_0_25+quarter_0_75)/2)
        df.loc[(df['Contract'] == 'Two Year') &(df['Total Charges'].isnull()),'Total Charges'] = str(quarter_0_75)
        df.loc[(df['Contract'] == 'Month-to-Month') &(df['Total Charges'].isnull()),'Total Charges'] = str(quarter_0_25)

        # check_isnull(df, 'Contract')
        # check_isnull(df, 'Total Charges')

        counter = 0
        record_bad_row = []
        for i in range(df.shape[0]):
            detect_bad_row = sum([pd.isnull(df.loc[i,"Total Charges"]),pd.isnull(df.loc[i,"Total Refunds"]),pd.isnull(df.loc[i,"Total Extra Data Charges"]),pd.isnull(df.loc[i,"Total Long Distance Charges"]),pd.isnull(df.loc[i,"Total Revenue"])])
            if detect_bad_row >=2:
                record_bad_row.append([i,detect_bad_row ])
                counter+=1

        # print(record_bad_row)
        Total_Charges_mean = df["Total Charges"].astype(float).mean()
        Total_Refunds_mean = df["Total Refunds"].astype(float).mean()
        Total_Extra_Data_Charges_mean = df["Total Extra Data Charges"].astype(float).mean()
        Total_Long_Distance_Charges_mean = df["Total Long Distance Charges"].astype(float).mean()
        Total_Revenue_mean = df["Total Revenue"].astype(float).mean()



        for bad_data in record_bad_row:
            row,num = bad_data[0],bad_data[1]
            counter = num
            if counter>1 and pd.isnull(df.loc[row,"Total Charges"]):
                df.loc[row, "Total Charges"] = Total_Charges_mean
                counter-=1

            if counter>1 and pd.isnull(df.loc[row,"Total Refunds"]):
                df.loc[row, "Total Refunds"] = Total_Refunds_mean
                counter -= 1
            if counter>1 and pd.isnull(df.loc[row,"Total Extra Data Charges"]):
                df.loc[row, "Total Extra Data Charges"] = Total_Extra_Data_Charges_mean
                counter -= 1
            if counter>1 and pd.isnull(df.loc[row,"Total Long Distance Charges"]):
                df.loc[row, "Total Long Distance Charges"] = Total_Long_Distance_Charges_mean
                counter -= 1
            if counter>1 and pd.isnull(df.loc[row,"Total Revenue"]):
                df.loc[row, "Total Revenue"] = Total_Revenue_mean
                counter -= 1
        # 經過以上處理，(Total Charge)-(Total Refunds)+(Total Extra Data Charges)+(Total Long Distance Charges) =Total Revenue  只會有一個未知數


        df["Total Charges"] = df["Total Charges"].astype(np.float32)
        df["Total Refunds"] = df["Total Refunds"].astype(np.float32)
        df["Total Extra Data Charges"] = df["Total Extra Data Charges"].astype(np.float32)
        df["Total Revenue"] = df["Total Revenue"].astype(np.float32)
        df["Total Long Distance Charges"] = df["Total Long Distance Charges"].astype(np.float32)


        for i in range(df.shape[0]):
            if pd.isnull(df.loc[i, "Total Charges"]):
                df.loc[i, "Total Charges"] = df.loc[i, "Total Revenue"]-df.loc[i, "Total Extra Data Charges"]-df.loc[i, "Total Long Distance Charges"]\
                                             +df.loc[row,"Total Refunds"]

            elif pd.isnull(df.loc[i, "Total Refunds"]):
                df.loc[i, "Total Refunds"] = df.loc[i, "Total Charges"]+df.loc[i, "Total Extra Data Charges"]+df.loc[i, "Total Long Distance Charges"]-df.loc[i,"Total Revenue"]


            elif pd.isnull(df.loc[i, "Total Extra Data Charges"]):
                df.loc[i, "Total Extra Data Charges"] = df.loc[i, "Total Revenue"]-df.loc[i, "Total Charges"]-df.loc[i, "Total Long Distance Charges"]+df.loc[i, "Total Refunds"]

            elif pd.isnull(df.loc[i, "Total Long Distance Charges"]):

                df.loc[i, "Total Long Distance Charges"] = df.loc[i, "Total Revenue"] - df.loc[i, "Total Charges"] - df.loc[i, "Total Extra Data Charges"] + df.loc[i, "Total Refunds"]

            elif pd.isnull(df.loc[i,"Total Revenue"]):
                df.loc[i, "Total Revenue"] = df.loc[i, "Total Extra Data Charges"]+df.loc[i, "Total Charges"]+df.loc[i, "Total Long Distance Charges"]-df.loc[i, "Total Refunds"]
        return df
    df = preprocess_data(combine_df,intresting_features)
    intresting_features.pop(intresting_features.index("Under 30"))
    intresting_features.pop(intresting_features.index("Lat Long"))
    combine_df = df[intresting_features]
    train_df = combine_df.loc[:2487]
    test_df = combine_df.loc[2488:]
    train_df.to_csv("./Final_csv/train_DATA.csv",index=False)
    test_df.to_csv("./Final_csv/test_DATA.csv",index=False)

# yes:no -> 1:0
# Senior Citizen、Referred a Friend、Phone Service、Multiple Lines、Internet Service、
# Online Security、Online Backup、Device Protection Plan、Premium Tech Support、Unlimited Data

# Internet Type:none、fiber optic、dsl、cable轉0、1、2、3
# Contract:month-to-month、one-year、two-year轉:0、1、2

def trans_data_to_01(data_path):
    p,f_name = os.path.split(data_path)
    df = pd.read_csv(data_path)
    yes_no_features = ['Senior Citizen','Referred a Friend','Phone Service','Multiple Lines','Internet Service','Online Security','Online Backup',
              'Device Protection Plan','Premium Tech Support','Unlimited Data']
    Internet_Type = ['Internet Type']
    Contract = ['Contract']

    for f in yes_no_features:
        df.loc[df[f] == "Yes",f]=1
        df.loc[df[f] == "No", f] = 0

    df.loc[df['Internet Type'] == 'None','Internet Type'] = 0
    df.loc[df['Internet Type'] == 'Fiber Optic','Internet Type'] = 1
    df.loc[df['Internet Type'] == 'DSL','Internet Type'] = 2
    df.loc[df['Internet Type'] == 'Cable','Internet Type'] = 3

    #'Month-to-Month'  One Year  Two Year
    df.loc[df['Contract'] == 'Month-to-Month', 'Contract'] = 0
    df.loc[df['Contract'] == 'One Year', 'Contract'] = 1
    df.loc[df['Contract'] == 'Two Year','Contract'] = 2

    # No Churn、 Other 、Price 、Dissatisfaction、 Attitude、Competitor、
    df.loc[df['Churn Category'] == 'No Churn','Churn Category'] = 0
    df.loc[df['Churn Category'] == 'Competitor','Churn Category'] = 1
    df.loc[df['Churn Category'] == 'Dissatisfaction','Churn Category'] = 2
    df.loc[df['Churn Category'] == 'Attitude','Churn Category'] = 3
    df.loc[df['Churn Category'] == 'Price','Churn Category'] = 4
    df.loc[df['Churn Category'] == 'Other','Churn Category'] = 5



    df.to_csv(f'./Final_csv/DATA_01/{f_name}',index = False)
# trans_data_to_01('./Final_csv/DATA_No_01/train_DATA.csv')


feature = ["./demographics.csv", "./location.csv", "./satisfaction.csv", "./services.csv"]
def use_nouse_train_ID(ori_train_id,aft_train_id,*args):
    intresting_features = ["Customer ID","Age","Under 30","Senior Citizen","Lat Long","Latitude","Longitude","Satisfaction Score",
                           "Referred a Friend","Number of Referrals","Tenure in Months","Phone Service",
                           "Multiple Lines","Internet Service","Internet Type","Online Security","Online Backup",
                           "Device Protection Plan","Premium Tech Support","Unlimited Data","Contract","Total Charges",
                           "Total Refunds","Total Extra Data Charges","Total Long Distance Charges","Total Revenue","Churn Category"]
    all_train_id = pd.read_csv("./Train_IDs.csv")
    use_id = pd.read_csv("./Final_csv/DATA_No_01/train_DATA.csv")
    all_id_list = all_train_id["Customer ID"].tolist()
    use_id_list = use_id["Customer ID"].tolist()

    # for i,all_id in enumerate(all_id_list):
    #     # 如果all_id是已經拿來train過的
    #     if all_id in use_id_list:
    #         all_id_list.pop(i)
    # print(len(all_id_list))

    for id in use_id_list:
        if id in all_id_list:
            idx = all_id_list.index(id)
            all_id_list.pop(idx)
    no_use_id_list = all_id_list
    print(len(all_id_list))
    # input()


    more_train_id = pd.DataFrame()
    # more_train_id["Customer ID"] = np.nan
    for id in no_use_id_list:
        id_concate = pd.DataFrame()
        id_concate["Customer ID"] = [id]
        for csv_file in args:
            df = pd.read_csv(csv_file)
            id_row = df["Customer ID"] == id
            if id_row.sum() == 1:
                customer_id = df.loc[id_row]
                customer_id = customer_id.drop("Customer ID",axis = 1)
            else:
                features = df.columns.tolist()
                val = [np.nan for _ in range(len(features))]
                customer_id = pd.DataFrame([val],columns=features)
                customer_id = customer_id.drop("Customer ID", axis=1)


            id_concate = id_concate.reset_index()
            customer_id = customer_id.reset_index()
            # id_concate.to_csv("./5566.csv",index = False)
            # customer_id.to_csv("./7788.csv",index = False)
            id_concate = pd.concat([id_concate,customer_id],axis=1)
            id_concate = id_concate.drop('index',axis = 1)
        more_train_id = more_train_id.append(id_concate,ignore_index=True)
    more_train_id.to_csv("./ans.csv",index=False)

# use_nouse_train_ID("./Final_csv/DATA_No_01/train_DATA.csv","./","./demographics.csv", "./location.csv", "./satisfaction.csv", "./services.csv",'./status.csv')


def fill_no_use_train_id(no_use_train_id_path,train_id_path):
    no_use_id = pd.read_csv(no_use_train_id_path)
    use_id = pd.read_csv(train_id_path)
    use_id_features = use_id.columns.tolist()
    no_use_id = no_use_id[use_id_features]
    no_use_id.dropna(subset = ['Churn Category'], inplace=True)
    no_use_id = no_use_id.reset_index()
    #以上把所有沒label的資料拔了，大概多了2千筆

    for f in no_use_id.columns.tolist():
        for i in range(no_use_id.shape[0]):
            if pd.isnull(no_use_id.loc[i,f]):
                label = no_use_id.loc[i,'Churn Category']
                candidate = no_use_id.loc[no_use_id['Churn Category'] == label,f]
                candidate.dropna(inplace=True)
                candidate = (candidate.index).tolist()
                final_choise = random.choice(candidate)
                no_use_id.loc[i,f] = no_use_id.loc[final_choise,f]
    no_use_id.to_csv("./Final_train_id.csv",index = False)
# fill_no_use_train_id("./ans.csv","./Final_csv/DATA_No_01/train_DATA.csv")
def concate_df():
    df1 = pd.read_csv('./Final_csv/DATA_No_01/train_DATA.csv')
    df2 = pd.read_csv('./Final_train_id.csv')
    df2 = df2[df1.columns]
    df = pd.concat([df1,df2],axis=0)
    l1 = df['Customer ID'].tolist()
    l2 = list(set(l1))
    df.to_csv("./all_fillna_train_id.csv",index = False)
def remove_some_category(df,want_to_remove):
    counter = 0
    for i,row in df.iterrows():
        if counter==want_to_remove:
            break
        if row['Churn Category'] ==0:
            df.loc[i,'Churn Category'] = np.nan
        counter+=1
    df = df.dropna(axis=0, how='any')
    df.reset_index()
    return df


def training_and_validate(train_data_path,test_path = "./Final_csv/DATA_01/test_DATA.csv"):

    df = pd.read_csv(train_data_path)
    # df = df.drop(columns=["Customer ID",'Senior Citizen','Referred a Friend','Age',"Total Charges",
    #                        "Total Refunds","Total Extra Data Charges","Total Long Distance Charges",])
    df = df.drop(columns=["Customer ID"])
    # df = remove_some_category(df,1000)


    # f = ['Senior Citizen', 'Referred a Friend', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security',
    #  'Online Backup','Device Protection Plan', 'Premium Tech Support', 'Unlimited Data']
    # print(len(train_df.columns))
    col = df.columns.tolist()
    features,label =col[:-1],[col[-1]]
    X_df = df[features]
    Y_df = df[label]
    # 去除outlier 約486筆
    # detector = EllipticEnvelope()
    # detector.fit(X_df)
    # a = detector.predict(X_df)
    # a[a==1] = 0
    # a[a==-1] = 1
    #
    # for i in range(len(a)):
    #     if a[i] == 1:
    #         a[i] = i
    # X_df.loc[a] = np.nan
    # X_df = X_df.dropna()
    # Y_df = df[label]
    # Y_df.loc[a] = np.nan
    # Y_df = Y_df.dropna()
    # 去除outlier 約486筆


    # kf = KFold(n_splits=3)
    # for train,test in kf.split(X_df):
    #     train_X,test_X,train_Y,test_Y = X_df.loc[train],X_df.loc[test],Y_df.loc[train],Y_df.loc[test]
    #     normallize = preprocessing.StandardScaler()
    #     train_X = normallize.fit_transform(train_X)
    #     val_X = normallize.fit_transform(val_X)

    train_X, val_X, train_y, val_y = model_selection.train_test_split(X_df, Y_df, test_size=0.1)

    # 資料標準化
    # normallize = preprocessing.StandardScaler()
    # train_X = normallize.fit_transform(train_X)
    # val_X = normallize.fit_transform(val_X)
    # 資料標準化

    # forest = ensemble.RandomForestClassifier(n_estimators=8,class_weight='balanced_subsample',max_features = 'auto')
    # print(cross_val_score(forest,X_df,Y_df,cv = 3))
    # input()
    # t = train_y['Churn Category'].tolist()
    # print("train_y :",Counter(t))
    # forest.fit(train_X, train_y)
    # val_pred = forest.predict(val_X)
    # print("RF result is ","\n",metrics.classification_report(val_y,val_pred))
    # print("val_pred :", Counter(t))



    # cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
    # scores = cross_val_score(forest, X_df, Y_df, scoring='roc_auc', cv=cv, n_jobs=-1)
    # print(scores)
    # input()
    # ,class_weight = {0: 1, 1: 2.5 , 2 : 2.5 , 3:2.5, 4:2.5, 5:2.5}
    # max_depth =10 ,

    from imblearn.ensemble import BalancedRandomForestClassifier
    model = BalancedRandomForestClassifier(n_estimators=3,max_features = 'auto',class_weight='balanced_subsample',max_depth=18)
    model.fit(train_X, train_y)
    val_pred = model.predict(val_X)


    print("IMB RF result is ", "\n", metrics.classification_report(val_y, val_pred))
    # input()

    df =pd.read_csv(test_path)
    # test_df = df.drop(columns=["Customer ID",'Churn Category','Senior Citizen','Referred a Friend',
    #                            'Age',"Total Charges","Total Refunds","Total Extra Data Charges","Total Long Distance Charges",])
    test_df = df.drop(columns=["Customer ID"])



    # test_pred = forest.predict(test_df)
    # print("forest res is : ",Counter(test_pred))

    test_pred_2 = model.predict(test_df)
    input()
    print("Y_df :",Counter(Y_df['Churn Category'].tolist()))
    print("IMB forest res is : ",Counter(test_pred_2))
    # print(type(test_pred_2))
    # test_df = pd.read_csv(test_path)
    res_df = df[["Customer ID",'Churn Category']]

    ID_dict = {}
    for i,row in res_df.iterrows():
        res_df.loc[i,"Churn Category"] = int(test_pred_2[i])
        id = row["Customer ID"]
        ID_dict[id] = test_pred_2[i]
    # print(ID_dict)

    test_ID_file = pd.read_csv("./Final_csv/Pred_res/Test_IDs.csv")
    for i,row in test_ID_file.iterrows():
        id = row["Customer ID"]
        test_ID_file.loc[i,'Churn Category'] = str(int(ID_dict[id]))
    t = test_ID_file['Churn Category'].tolist()
    print(t)
    input()
    print(Counter(t))
    input()
    test_ID_file.to_csv("./Final_csv/Pred_res/res.csv",index = False)

training_and_validate("./Final_csv/DATA_01/all_fillna_train_id.csv")

if __name__ == '__main__':
    pass
    # status.csv就是label的意思，那麼Test_ID當然不會在status.csv裡面
    # feature = ["demographics.csv", "location.csv", "satisfaction.csv", "services.csv"]
    # best_id, bad_id = check_ID_data_Integrity("./Test_IDs.csv", feature)

    # print(len(bad_id))
    # print(len(best_id))
    # input()



    # train_Data_path = "./train_ID_ver1.csv"
    # test_Data_path = "./test_ID_ver1.csv"
    # DATA_Preprocess(train_Data_path,test_Data_path)


    # total_csv_path = "./train_data_ver1.csv"
    # train_id_path = "./Train_IDs.csv"
    # test_id_path = "./Test_IDs.csv"
    # split_train_test(total_csv_path,train_id_path,test_id_path)

    # 以下將所有資料分割且填值且轉換成one-hot encoding
    # test_path = "./Final_csv/DATA_No_01/test_DATA.csv"
    # train_path = "./Final_csv/DATA_No_01/train_DATA.csv"
    # f_list =test_path,train_path
    # for f in f_list:
    #     trans_data_to_01(f)




