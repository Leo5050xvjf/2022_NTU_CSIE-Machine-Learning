import numpy as np
from numpy.linalg import multi_dot



# 1.取出所有(training data,training labels) ， (testing data,testing labels)
# 2.將所有資料(both test and train)分別轉換至Q12~Q16指定的domain ，
# 3.計算 M_plus = np.linalg.pinv
# 4.W_lin = np.dot(M_plus,labels)
# 5.|計算err_in(g) - err_out(g)|

# M_plus = np.linalg.pinv(DataSET[0][0])
# W_lin = np.dot(M_plus, DataSET[0][1])

# Q12 Q13
def Q_order(data,order):

    len_data = len(data)
    ans = np.zeros(1+len_data*order)
    temp_ans = np.zeros((order,len_data))
    for _ in range(1,order+1):
        temp_ans[_-1] = data ** _
    temp_ans = temp_ans.reshape(-1)
    ans[0] = 1
    ans[1:] = temp_ans
    return ans
# Q14
def full_order_2_polynomial(data):
    len_data = len(data)

    second_order = []
    counter = 0
    for i in range(len_data):
        x1 = data[i]
        for j in range(counter,len_data):
            second_order.append(x1*data[j])
        counter += 1
    data = np.insert(data,0,1,axis=0)
    data =list(data)
    t_data = np.array(data+second_order)
    return t_data
# Q15
def trans_lower_dim(data,split_num  = 1):
    data = data[:split_num]
    data = np.insert(data,0,1,axis=0)
    return data
def random_Dimension(data):
    data = np.array(data)
    index = np.random.choice(np.arange(10),size = 5,replace=False)
    data = data[index]
    data = list(data)
    x_0 = [1]
    data = np.array(x_0+data)
    return data
def read_data_and_trans(data_path,transTo,Q15_dim = 1)->np.ndarray:
    All_data = []
    with open(data_path) as file:
        for data in file:
            data = [float(d) for d in data.split()]
            All_data.append(data)
    All_data = np.array(All_data)
    data,labels = All_data[:,:-1],All_data[:,-1:]
    labels = labels.reshape(-1)
    h,w = data.shape
    transData = []
    if transTo == "Q12":
        # trans_data = np.zeros((h,w),dtype=np.float32)
        for _ in range(h):
            t_data = Q_order(data[_],2)
            transData.append(t_data)
        transData = np.array(transData)
    if transTo == "Q13":
        for _ in range(h):
            t_data = Q_order(data[_],8)
            transData.append(t_data)
        transData = np.array(transData)
    if transTo == "Q14":
        for _ in range(h):
            t_data = full_order_2_polynomial(data[_])
            transData.append(t_data)
        transData = np.array(transData)
    if transTo == "Q15":
        for _ in range(h):
            t_data = trans_lower_dim(data[_],Q15_dim)
            transData.append(t_data)
        transData = np.array(transData)
    if transTo == "Q16":
        for _ in range(h):
            t_data = random_Dimension(data[_])
            transData.append(t_data)
        transData = np.array(transData)
    return transData,labels
def Q12(training_data_pathh,testing_data_path):
    t_traning_data ,training_labels = read_data_and_trans(training_data_path,"Q12",Q15_dim=0)
    t_testinging_data ,testing_labels = read_data_and_trans(testing_data_path,"Q12",Q15_dim=0)

    M_plus = np.linalg.pinv(t_traning_data)
    W_lin = np.dot(M_plus, training_labels).reshape(21,1)

    tran_pred = np.dot(t_traning_data,W_lin)
    tran_pred[tran_pred>=0] = 1
    tran_pred[tran_pred<0] = -1
    tran_pred = tran_pred.reshape(-1)

    pred = np.dot(t_testinging_data,W_lin)
    pred[pred>=0] = 1
    pred[pred<0] = -1
    pred = pred.reshape(-1)

    Ein = sum(tran_pred==training_labels) / len(tran_pred)
    Eout  = sum(pred == testing_labels) / len(pred)

    return abs(Ein- Eout)
def Q13(training_data_path,testing_data_path):
    t_traning_data, training_labels = read_data_and_trans(training_data_path, "Q13", Q15_dim=0)
    t_testinging_data, testing_labels = read_data_and_trans(testing_data_path, "Q13", Q15_dim=0)

    M_plus = np.linalg.pinv(t_traning_data)
    W_lin = np.dot(M_plus, training_labels).reshape(81, 1)

    tran_pred = np.dot(t_traning_data, W_lin)
    tran_pred[tran_pred >= 0] = 1
    tran_pred[tran_pred < 0] = -1
    tran_pred = tran_pred.reshape(-1)

    pred = np.dot(t_testinging_data, W_lin)
    pred[pred >= 0] = 1
    pred[pred < 0] = -1
    pred = pred.reshape(-1)

    Ein = sum(tran_pred == training_labels) / len(tran_pred)
    Eout = sum(pred == testing_labels) / len(pred)

    return abs(Ein - Eout)
def Q14(training_data_path,testing_data_path):
    t_traning_data, training_labels = read_data_and_trans(training_data_path, "Q14", Q15_dim=0)
    t_testinging_data, testing_labels = read_data_and_trans(testing_data_path, "Q14", Q15_dim=0)

    t_testinging_data_len = len(t_testinging_data[0])

    M_plus = np.linalg.pinv(t_traning_data)
    W_lin = np.dot(M_plus, training_labels).reshape(t_testinging_data_len, 1)

    tran_pred = np.dot(t_traning_data, W_lin)
    tran_pred[tran_pred >= 0] = 1
    tran_pred[tran_pred < 0] = -1
    tran_pred = tran_pred.reshape(-1)

    pred = np.dot(t_testinging_data, W_lin)
    pred[pred >= 0] = 1
    pred[pred < 0] = -1
    pred = pred.reshape(-1)

    Ein = sum(tran_pred == training_labels) / len(tran_pred)
    Eout = sum(pred == testing_labels) / len(pred)

    return abs(Ein - Eout)
def Q15(training_data_path,testing_data_path):
    # 假設一個全錯，一個全對，取abs最大也就2
    find_min_err_i = 2
    Err_bag = []
    ans = 0
    for i in range(1,10+1):
        t_traning_data, training_labels = read_data_and_trans(training_data_path, "Q15", Q15_dim=i)
        t_testinging_data, testing_labels = read_data_and_trans(testing_data_path, "Q15", Q15_dim=i)

        t_testinging_data_len = len(t_testinging_data[0])

        M_plus = np.linalg.pinv(t_traning_data)
        W_lin = np.dot(M_plus, training_labels).reshape(t_testinging_data_len, 1)

        tran_pred = np.dot(t_traning_data, W_lin)
        tran_pred[tran_pred >= 0] = 1
        tran_pred[tran_pred < 0] = -1
        tran_pred = tran_pred.reshape(-1)

        pred = np.dot(t_testinging_data, W_lin)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        pred = pred.reshape(-1)

        Ein = sum(tran_pred == training_labels) / len(tran_pred)
        Eout = sum(pred == testing_labels) / len(pred)
        Err_bag.append(abs(Ein - Eout))
        if abs(Ein - Eout) < find_min_err_i:
            find_min_err_i = abs(Ein - Eout)
            ans = i


    return Err_bag,ans
def Q16(training_data_path,testing_data_path):

    average_err = 0
    for _ in range(200):
        t_traning_data, training_labels = read_data_and_trans(training_data_path, "Q16", Q15_dim=0)
        t_testinging_data, testing_labels = read_data_and_trans(testing_data_path, "Q16", Q15_dim=0)
        t_testinging_data_len = len(t_testinging_data[0])

        M_plus = np.linalg.pinv(t_traning_data)
        W_lin = np.dot(M_plus, training_labels).reshape(t_testinging_data_len, 1)

        tran_pred = np.dot(t_traning_data, W_lin)
        tran_pred[tran_pred >= 0] = 1
        tran_pred[tran_pred < 0] = -1
        tran_pred = tran_pred.reshape(-1)

        pred = np.dot(t_testinging_data, W_lin)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        pred = pred.reshape(-1)

        Ein = sum(tran_pred == training_labels) / len(tran_pred)
        Eout = sum(pred == testing_labels) / len(pred)
        average_err += abs(Ein - Eout)
    average_err = average_err/200
    return average_err


if __name__ == '__main__':

    training_data_path = "./hw3_train.txt"
    testing_data_path = "./hw3_test.txt"
    Q12_ans = Q12(training_data_path, testing_data_path)
    print(f"Q12 : {Q12_ans}")

    Q13_ans = Q13(training_data_path, testing_data_path)
    print(f"Q13 : {Q13_ans}")

    Q14_ans = Q14(training_data_path, testing_data_path)
    print(f"Q14 : {Q14_ans}")

    all_err, min_err_index = Q15(training_data_path, testing_data_path)
    # print(all_err)
    print(f"Q15 : {min_err_index}")

    Q16_ans = Q16(training_data_path, testing_data_path)
    print("Q16 : ", Q16_ans)



