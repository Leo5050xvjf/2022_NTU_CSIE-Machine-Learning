
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from liblinear.liblinearutil import *



def third_order_polytrans(datapath):

    data = np.loadtxt(datapath)
    x,y = data[:,:-1],data[:,-1]
    trans = PolynomialFeatures(degree=3)
    x = trans.fit_transform(x)
    y = list(map(int,y))
    data_bag = []
    for data in x:
        dict_ = {}
        for key,val in enumerate(data,1):
            dict_[key] = val
        data_bag.append(dict_)
    return y,data_bag


def Q12(lambd):

    acc_bag = []
    for i in range(5):
        model = train(test_y,test_data,f"-s 0 -c {1/(2*lambd[i])} -e 0.000001 -q")
        p_label, p_acc, p_test = predict(test_y, test_data, model)
        acc_bag.append(p_acc[0])
    return acc_bag

def Q13(lambd):
    acc_bag = []
    for i in range(5):
        model = train(tra_y,tra_data,f"-s 0 -c {1/(2*lambd[i])} -e 0.000001 -q")
        p_label, p_acc, p_test = predict(tra_y, tra_data, model)
        acc_bag.append(p_acc[0])
    return acc_bag
def Q14(lambd,tra_data,tra_y,test_data,test_y):
    acc_bag = []
    tra_d,val_d = tra_data[:120],tra_data[120:]
    tra_y ,val_y = tra_y[:120],tra_y[120:]
    for i in range(5):
        model = train(tra_y,tra_d,f"-s 0 -c {1/(2*lambd[i])} -e 0.000001 -q")
        p_label, p_acc, p_test = predict(val_y, val_d, model)
        acc_bag.append(p_acc[0])
    
    # find best and last index lambda
    best_lambda = 0
    best_acc = acc_bag[0]
    for i,_ in enumerate(acc_bag):
        if _ >= best_acc:
            best_acc = _
            best_lambda  = i
    print("===================================testing=======================================")
    model = train(tra_y, tra_d, f"-s 0 -c {1 / (2 * lambd[best_lambda])} -e 0.000001 -q")
    p_label, p_acc, p_test = predict(test_y, test_data, model)
    return best_lambda,p_acc[0]

def Q15(lambd,tra_data,tra_y,test_data,test_y):
    # Q14的best_lambda 是 lambd[3]
    model = train(tra_y,tra_data,f"-s 0 -c {1/(2*lambd[3])} -e 0.000001 -q")
    p_label, p_acc, p_test = predict(test_y, test_data, model)
    return p_acc[0]
def Q16(lambd,tra_data,tra_y):
    split_data = [tra_data[:40],tra_data[40:80],tra_data[80:120],tra_data[120:160],tra_data[160:]]
    split_y = [tra_y[:40],tra_y[40:80],tra_y[80:120],tra_y[120:160],tra_y[160:]]
    average_acc_bag = []

    for l in range(5):
        acc = 0
        for i in range(5):
            val_data = split_data[i]
            val_y = split_y[i]
            tra_d = []
            tra_y = []
            for j in range(5):
                if j == i:pass
                else:
                    tra_d+=split_data[j]
                    tra_y+=split_y[j]

            model = train(tra_y, tra_d, f"-s 0 -c {1 / (2 * lambd[l])} -e 0.000001 -q")
            p_label, p_acc, p_test = predict(val_y, val_data , model)
            acc+=p_acc[0]
        average_acc_bag.append(acc/5)
    return average_acc_bag

if __name__ == '__main__':
    tra_datapath = "./hw4_train.dat.txt"
    tra_y, tra_data = third_order_polytrans(tra_datapath)
    test_datapath = "./hw4_test.dat.txt"
    test_y, test_data = third_order_polytrans(test_datapath)

    log_lambd = [-4, -2, 0, 2, 4]
    lambd = [10 ** i for i in log_lambd]
    
    Q12_acc = Q12(lambd)
    Q13_acc = Q13(lambd)
    Q14_lambda,Q14_acc = Q14(lambd,tra_data,tra_y,test_data,test_y)
    Q15_acc = Q15(lambd,tra_data,tra_y,test_data,test_y)
    Q16_acc = Q16(lambd,tra_data,tra_y)








