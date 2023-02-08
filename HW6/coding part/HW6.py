
import numpy as np
import math
from tqdm import tqdm






# step1: 將所有資料根據feature 排序，目前的想法是 [排序後的feature1 : label],[排序後的feature2 : label]...
# step2: 根據每個feature iter 500 次，找出最適合的[s,theta]，
# step3: by step1 、step2 10個features 會產生對應的10個g_t (也就是10組(s,theta))
# step4: 將這10組g_t組合 ，即是G_t



def creatSortedData(dataPath):
    All_data = []
    with open(dataPath) as file:
        for data in file:
            data = [float(d) for d in data.split()]
            All_data.append(data)
    All_data = np.array(All_data)
    data,label = All_data[:,:-1],All_data[:,-1:]

    sorted_feature_bag = []
    non_sorted_feature_bag = []
    for i in range(10):
        feature =data[:,i:i+1]
        feature_label = np.concatenate((feature,label),axis=1)

        feature_label = list(map(list,feature_label))
        non_sorted_feature_bag.append(feature_label)
        feature_label = sorted(feature_label)
        sorted_feature_bag.append(feature_label)

    return sorted_feature_bag,non_sorted_feature_bag



dataPath = "hw6_train.dat.txt"
sorted_feature,non_sorted_feature= creatSortedData(dataPath)

def get_s_theta_alpha(sorted_feature:list,non_sorted_feature,iter_num):

    init_u = np.array([1 / 1000 for _ in range(1000)])

    # 共iter_num個[all 小g , alpha_t]
    all_g_s_theta = []
    for _ in tqdm(range(iter_num)):
        s_theta_bag = []
        for i in range(10):
            feature = sorted_feature[i]
            feature = np.array(feature)
            sorted_data, sorted_label = feature[:, :-1].reshape(-1), feature[:, -1:].reshape(-1)
            s_list = [-1, 1]
            theta_list = [((feature[j][0] + feature[j + 1][0]) / 2) for j in range(len(feature) - 1)]
            # 此時的err是weighted err 而Q11 Q12 是0/1 err
            err = float("inf")
            for s in s_list:
                for theta in theta_list:
                    data_bool = sorted_data - theta
                    data_bool[data_bool >= 0] = 1
                    data_bool[data_bool < 0] = -1
                    data_bool = data_bool * s
                    # 以上得到h_s_i_theta
                    bool_y_n_g_t = data_bool != sorted_label

                    E_in = np.dot(bool_y_n_g_t,init_u)
                    if E_in<err:
                        err = E_in
                        temp_s_theta = (s,theta)
            #此時某feature 的 小g已經找到最佳(s,theta)
            s_theta_bag.append((s,theta))
        #此時已獲得第t次iter 的所有小g，搭配u之後即可組成大G

        vote_arr = np.zeros(1000)
        for k in range(10):
            feature = non_sorted_feature[i]
            feature = np.array(feature)
            feature_i, label = feature[:, :-1].reshape(-1), feature[:, -1:].reshape(-1)
            s,theta = s_theta_bag[k]

            feature_i = feature_i-theta
            feature_i[feature_i>=0] = 1
            feature_i[feature_i<0] = -1
            predLabel = feature_i*s
            # feature_weight_vote = np.multiply(init_u,predLabel)
            vote_arr+=predLabel
        vote_arr[vote_arr>=0] = 1
        vote_arr[vote_arr<0] = -1
        g_pred_res = vote_arr!=label

        epsilon_t = np.dot(init_u,g_pred_res)/np.sum(init_u)
        block_t = ((1-epsilon_t)/epsilon_t)**0.5
        alpha_t = math.log(block_t)

        all_g_s_theta.append([s_theta_bag,alpha_t])

        scale_arr =np.zeros(1000)
        scale_block_t = np.argwhere(g_pred_res == True)
        div_block_t = np.argwhere(g_pred_res == False)
        scale_arr[scale_block_t] = block_t
        scale_arr[div_block_t] = 1/block_t

        # 更新u_t->u_t+1
        # 錯誤的data放大
        # 正確的縮小
        init_u = np.multiply(init_u,scale_arr)

    return all_g_s_theta


def Q11(non_sorted_data,s_theta_bag,iter_num):
    G_vote_arr = np.zeros(1000)
    for _ in range(iter_num):
        g_vote_arr = np.zeros(1000)
        for i in range(10):
            feature = non_sorted_data[i]
            feature = np.array(feature)
            feature_i, label = feature[:, :-1].reshape(-1), feature[:, -1:].reshape(-1)

            s,theta = s_theta_bag[_][0][i]
            alpha_t = s_theta_bag[_][1]

            feature_i = feature_i-theta
            feature_i[feature_i>=0] = 1
            feature_i[feature_i <0] = -1
            feature_i = feature_i*s
            g_vote_arr+=feature_i
        g_vote_arr[g_vote_arr>=0] = 1
        g_vote_arr[g_vote_arr<0] = -1
        G_vote_arr+=g_vote_arr*alpha_t
    G_vote_arr[G_vote_arr>=0] = 1
    G_vote_arr[G_vote_arr<0] = -1

    acc = G_vote_arr == label
    acc = np.sum(acc)/1000
    err = 1-acc
    return acc,err






all_g_s_theta = get_s_theta_alpha(sorted_feature,non_sorted_feature,20)
Q11_ans = Q11(non_sorted_feature,all_g_s_theta,20)
print(Q11_ans)








































