# import liblinear
import numpy as np
from libsvm.svmutil import *
import scipy
from scipy.sparse import dok_matrix
from sys import getsizeof
import random
from tqdm import tqdm



# y, x = svm_read_problem('./satimage.scale_test.txt')
# set train mode
'''

-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	4 -- precomputed kernel (kernel values in training_set_file)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)
'''
def multiclass_to_binary(labels,target_num : int):
    # try to trans non target num's label to -1
    # and trans targets num's label to +1

    labels = np.array(labels)
    labels[labels != target_num] = -1.0
    labels[labels == target_num] = 1.0
    pos = np.where(labels == 1)
    labels = list(labels)
    return labels


def Q11(y,x,target_num,C):
    y = multiclass_to_binary(y,target_num)
    m = svm_train(y, x, f'-s 0 -t 0 -c {C} -q')
    mul_alpha_y = m.get_sv_coef()
    SVs = m.get_SV()
    SVs_num = len(SVs)
    coef = dok_matrix((1, SVs_num), dtype=np.float32)
    vecters = dok_matrix((60, SVs_num), dtype=np.float32)

    for i in range(SVs_num):
        coef[0,i] = mul_alpha_y[i][0]

    for i,dict_ in enumerate(SVs):
        for k,v in dict_.items():
            vecters[k,i] = v
    inner_dot = vecters.dot(scipy.transpose(coef))
    W = (scipy.transpose(inner_dot).dot(inner_dot) )[0,0] **0.5
    return W
def Q12(train_y, train_x,C,degree,coef,gamma):

    Q12_acc_bag = []
    Q13_SVs_num = []

    for target_num in [2,3,4,5,6]:
        train_y_temp = multiclass_to_binary(train_y, target_num)
        m = svm_train(train_y_temp, train_x, f'-s 0 -t 1 -c {C} -d {degree} -r {coef} -g {gamma} -q' )
        SVs = m.get_SV()
        Q13_SVs_num.append(len(SVs))
        # test_y = multiclass_to_binary(test_y, target_num)
        p_label, p_acc, p_val = svm_predict(train_y_temp, train_x, m)
        Q12_acc_bag.append(p_acc[0])
    return Q12_acc_bag,Q13_SVs_num
def Q14(target_num,train_y,train_x,test_y,test_x,gamma):
    C_list = [0.01,0.1,1,10,100]
    train_y_temp = multiclass_to_binary(train_y, target_num)
    test_y_temp = multiclass_to_binary(test_y,target_num)
    Q14_acc_bag = []
    for C in C_list:
        m = svm_train(train_y_temp, train_x, f'-s 0 -t 2 -c {C} -g {gamma} -q')
        p_label, p_acc, p_val = svm_predict(test_y_temp, test_x, m)
        Q14_acc_bag.append(p_acc[0])
    return Q14_acc_bag
# print(Q14_ans)

def Q15(target_num,train_y,train_x,test_y,test_x,C):
    gamma_list = [0.1,1,10,100,1000]
    train_y_temp = multiclass_to_binary(train_y, target_num)
    test_y_temp = multiclass_to_binary(test_y,target_num)
    Q15_acc_bag = []
    for gamma in gamma_list:
        m = svm_train(train_y_temp, train_x, f'-s 0 -t 2 -c {C} -g {gamma} -q')
        p_label, p_acc, p_val = svm_predict(test_y_temp, test_x, m)
        Q15_acc_bag.append(p_acc[0])
    return Q15_acc_bag
# print(Q15_ans)
def Q16(target_num,train_y,train_x,C = 0.1):

    gamma_list = [0.1,1,10,100,1000]
    train_y= multiclass_to_binary(train_y, target_num)
    vote_list = [0,0,0,0,0]
    for i in tqdm(range(1000)):

        acc_bag = []
        randon_sample = random.sample(range(len(train_y)), 200)
        train_y_temp = np.array(train_y)
        train_x_temp = np.array(train_x)
        val_x = list(train_x_temp[randon_sample])
        val_y = list(train_y_temp[randon_sample])
        train_x_temp = list(np.delete(train_x_temp, randon_sample))
        train_y_temp = list(np.delete(train_y_temp, randon_sample))
        for gamma in gamma_list:
            m = svm_train(train_y_temp, train_x_temp, f'-s 0 -t 2 -c {C} -g {gamma} -q')
            p_label, p_acc, p_val = svm_predict(val_y, val_x, m)
            acc_bag.append(p_acc[0])
        best_E_val = acc_bag.index(max(acc_bag))
        vote_list[best_E_val] += 1
    return vote_list

if __name__ == '__main__':
    train_y, train_x = svm_read_problem('./satimage.scale_train.txt')
    test_y, test_x = svm_read_problem("./satimage.scale_test.txt")
    Q11_ans = Q11(train_y,train_x,5,C=10)
    Q12_ans,Q13_ans = Q12(train_y, train_x,10,3,1,1)
    Q14_ans = Q14(1,train_y,train_x,test_y,test_x,10)
    Q15_ans = Q15(1,train_y,train_x,test_y,test_x,0.1)
    Q16_ans = Q16(1,train_y,train_x,0.1)
    print(f"Q11_ans{Q11_ans}")
    print(f"Q12_ans{Q12_ans}")
    print(f"Q13_ans{Q13_ans}")
    print(f"Q14_ans{Q14_ans}")
    print(f"Q15_ans{Q15_ans}")
    print(f"Q16_ans{Q16_ans}")























