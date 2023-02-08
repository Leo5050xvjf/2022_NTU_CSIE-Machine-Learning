
import numpy as np
import math
from tqdm import tqdm


def Theta_feature_label(dataPath):
    All_data = []
    with open(dataPath) as file:
        for data in file:
            data = [float(d) for d in data.split()]
            All_data.append(data)
    All_data = np.array(All_data)
    data,label = All_data[:,:-1],All_data[:,-1:]
    label = label.reshape(-1).tolist()


    feature_bag = []
    feature_theta_bag = []
    theta_bag = []
    for i in range(10):
        theta_i_bag = []

        feature =data[:,i:i+1].reshape(-1)
        feature =feature.tolist()
        feature_bag.append(feature)
        # feature_label = np.concatenate((feature,label),axis=1)
        sorted_feature = sorted(feature)

        for j in range(len(sorted_feature)-1):
            theta = (sorted_feature[j]+sorted_feature[j+1])/2
            theta_i_bag.append(theta)
        theta_bag.append(theta_i_bag)
    return feature_bag,label,theta_bag
def get_all_g(feature_bag,label,theta_bag,iter_num):
    numOfData = len(label)
    init_u = np.array([1 / numOfData for _ in range(numOfData)])
    label =np.array(label)
    total_g_alpha = []
    for _ in tqdm(range(iter_num)):
        Ein =float("inf")
        s_list = [-1,1]
        for i in range(10):
            feature_i = feature_bag[i]
            feature_i = np.array(feature_i)
            theta_i =theta_bag[i]
            for s in s_list:
                for theta in theta_i:
                    feature_i_pred = feature_i-theta
                    feature_i_pred[feature_i_pred>=0] = 1
                    feature_i_pred[feature_i_pred<0] = -1
                    feature_i_pred*=s
                    #以上得到h(X)
                    pred_res = feature_i_pred!=label
                    err =np.dot(init_u,pred_res)/numOfData

                    if err < Ein:
                        Ein = err
                        temp_best_s_i_theta = (s,i,theta)
                        yn_g_bool = pred_res

        epsilon_t = np.dot(init_u,yn_g_bool)/np.sum(init_u)
        block_t = ((1-epsilon_t)/epsilon_t) **0.5
        alpha_t = math.log(block_t)
        total_g_alpha.append([alpha_t,temp_best_s_i_theta])

        #update u_t->u_t+1
        scale_table = np.zeros(numOfData)
        # True 代表不等label的部分
        scale_table[yn_g_bool==True] = block_t
        scale_table[yn_g_bool==False] = 1/block_t
        init_u =np.multiply(init_u,scale_table)

    return total_g_alpha
# numOfT :決定要用幾個小g來算結果
def Big_G(feature_bag,label,total_g_alpha,numOfT,alpha_bool = 1):

    numOfData = len(label)
    vote_arr = np.zeros(numOfData)
    all_iter_err = []
    Q13_index = -1
    for T in range(numOfT):
        alpha_t = total_g_alpha[T][0]
        (s,i,theta) = total_g_alpha[T][1]
        # 如果不想要權重，alpha_t=1
        if alpha_bool!=1:
            alpha_t=1

        feature_i = feature_bag[i]
        feature_i = np.array(feature_i)
        pred_feature = feature_i-theta
        pred_feature[pred_feature>=0] = 1
        pred_feature[pred_feature<0] = -1
        pred_feature*=s

        p = pred_feature!=label
        g_err = np.sum(p)/numOfData
        all_iter_err.append(g_err)

        pred_feature*=alpha_t

        vote_arr+=pred_feature

        temp_pred = np.zeros(numOfData)
        temp_pred[vote_arr>=0] = 1
        temp_pred[vote_arr<0] = -1
        temp_err= np.sum(temp_pred!=label)/numOfData
        if temp_err<=0.05 and Q13_index==-1 :
            Q13_index = T


    vote_arr[vote_arr>=0] = 1
    vote_arr[vote_arr<0] = -1
    acc = np.sum(vote_arr==label)/numOfData
    err = 1-acc

    return acc,err,all_iter_err,Q13_index


def Q11(total_g,train_feature_bag,train_label,iter_num):
    Q11_acc,Q11_err,_,_  =Big_G(train_feature_bag,train_label, total_g, iter_num, 0)
    return Q11_err
def Q12(total_g,train_feature_bag,train_label,iter_num):
    _, _,all_err, _ = Big_G(train_feature_bag, train_label, total_g, iter_num, 0)
    id = all_err.index(max(all_err))
    print(f"id is {id}")
    print(f" alpha is {total_g[id][0]}")
    print(f" s,theta is {total_g[id][1]}")
    return max(all_err)
def Q13(total_g,train_feature_bag,train_label,iter_num):
    _, _, _, Q13_ans = Big_G(train_feature_bag, train_label, total_g, iter_num, 1)
    return Q13_ans
def Q14(total_g,test_feature_bag,test_label,iter_num):
    _, Q14_err, _, _ = Big_G(test_feature_bag, test_label, total_g, iter_num, 1)
    return Q14_err

def Q15(total_g,test_feature_bag,test_label,iter_num):
    _, Q_15_err, _, _ = Big_G(test_feature_bag, test_label, total_g, iter_num, 0)
    return Q_15_err
def Q16(total_g,test_feature_bag,test_label,iter_num):
    _, Q_16_err, _, _ = Big_G(test_feature_bag, test_label, total_g, iter_num, 1)
    return Q_16_err



if __name__ == '__main__':
    tra_dataPath = "hw6_train.dat.txt"
    val_dataPath = "hw6_test.dat.txt"
    train_feature_bag, train_label, theta_bag = Theta_feature_label(tra_dataPath)
    iter_num = 500
    # 等同於訓練在train data
    total_g = get_all_g(train_feature_bag, train_label, theta_bag, iter_num)
    val_feature_bag, val_label, theta_bag = Theta_feature_label(val_dataPath)

    Q11_err = Q11(total_g,train_feature_bag,train_label,1)
    Q12_err = Q12(total_g,train_feature_bag,train_label,500)
    Q13_err = Q13(total_g,train_feature_bag,train_label,500)
    Q14_err = Q14(total_g,val_feature_bag,val_label,1)
    Q15_err = Q15(total_g,val_feature_bag,val_label,500)
    Q16_err = Q16(total_g,val_feature_bag,val_label,500)
    print(Q11_err)
    print(Q12_err)
    print(Q13_err)
    print(Q14_err)
    print(Q15_err)
    print(Q16_err)












