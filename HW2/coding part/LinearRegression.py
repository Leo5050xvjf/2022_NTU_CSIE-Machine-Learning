import numpy as np
from numpy.linalg import multi_dot




def generateData(traNUM,valNUM,addOutliers = 0):

    # 產生200筆 training data traVal
    # 產生5000筆 testing data testVal
    traLabel = np.random.normal(0, 1, traNUM).reshape(traNUM,1)
    valLabel = np.random.normal(0, 1, valNUM).reshape(valNUM,1)

    # traData = np.zeros((200,2))
    # valData = np.zeros((5000,2))
    traData = []
    valData = []
    for _ in traLabel:
        if _[0] >= 0:
            #     產生 var = 0.6 ,mean  = [2 , 3]
            x1 = np.random.normal(2, 0.6 ** 0.5,1)[0]
            x2 = np.random.normal(3, 0.6 ** 0.5,1)[0]

        else:
            # 產生var = 0.4 ,mean  = [0 , 4]
            x1 = np.random.normal(0, 0.4 ** 0.5,1)[0]
            x2 = np.random.normal(4, 0.4 ** 0.5,1)[0]
        traData.append([1,x1, x2])
    for _ in valLabel:
        if _[0] >= 0:
            #     產生 var = 0.6 ,mean  = [2 , 3]
            x1 = np.random.normal(2, 0.6 ** 0.5,1)[0]
            x2 = np.random.normal(3, 0.6 ** 0.5,1)[0]
        else:
            # 產生var = 0.4 ,mean  = [0 , 4]
            x1 = np.random.normal(0, 0.4 ** 0.5,1)[0]
            x2 = np.random.normal(4, 0.4 ** 0.5,1)[0]
        valData.append([1,x1, x2])
    traData = np.asarray(traData,dtype=np.float64)
    valData = np.asarray(valData,dtype=np.float64)

    if addOutliers :
        Noise = []
        NoiseLabel = [1 for _ in range(20)]
        for _ in range(20):
            x1 = np.random.normal(6, 0.3 ** 0.5,1)[0]
            x2 = np.random.normal(0, 0.1 ** 0.5,1)[0]
            Noise.append([1,x1,x2])

        Noise = np.asarray(Noise,dtype=np.float64)
        NoiseLabel = np.asarray(NoiseLabel,dtype=np.float64).reshape(20,1)

        traData = np.concatenate((traData,Noise),axis=0)
        traLabel = np.concatenate((traLabel,NoiseLabel),axis=0)

    traLabel[traLabel >= 0] = 1
    traLabel[traLabel < 0] = -1
    valLabel[valLabel >= 0] = 1
    valLabel[valLabel < 0] = -1

    return [[traData,traLabel],[valData,valLabel]]


def meanErr(iterNUM,noiseTF = 0):
    lin_errinBag = []
    lin_errInOutBag = []
    lin_log_errout01Bag = []
    lin_log_errout01Bag_noise = []

    for _ in range(iterNUM):

        DataSET = generateData(200, 5000, 0)
        M_plus = np.linalg.pinv(DataSET[0][0])
        W_lin = np.dot(M_plus, DataSET[0][1])
        errin = np.dot(DataSET[0][0], W_lin) - DataSET[0][1]
        sqrErrMean = np.sum(errin ** 2) / len(errin)
        lin_errinBag.append(sqrErrMean)

        #for Q14
        lin_errin01  = linValidate(W_lin,DataSET[0][0],DataSET[0][1])
        lin_errout01 = linValidate(W_lin,DataSET[1][0],DataSET[1][1])
        lin_errInOutBag.append(abs(lin_errout01-lin_errin01))

        #for Q15
        log_errout01= logisticGrad(500,DataSET[0][0],DataSET[0][1],DataSET[1][0],DataSET[1][1])
        lin_log_errout01Bag.append([lin_errout01,log_errout01])
        #for Q16
        if noiseTF == 1:

            DataSET = generateData(200, 5000, 1)
            M_plus = np.linalg.pinv(DataSET[0][0])
            W_lin = np.dot(M_plus, DataSET[0][1])

            lin_errout01 = linValidate(W_lin, DataSET[1][0], DataSET[1][1])
            log_errout01= logisticGrad(500,DataSET[0][0],DataSET[0][1],DataSET[1][0],DataSET[1][1])
            lin_log_errout01Bag_noise.append([lin_errout01,log_errout01])


    lin_errIn01 = sum(lin_errinBag) / iterNUM
    lin_errInOutDiff01 = sum(lin_errInOutBag)[0] / iterNUM
    lin_log_errout01Bag = np.array(lin_log_errout01Bag)
    lin_log_err = np.mean(lin_log_errout01Bag,axis=0).reshape(-1)

    lin_log_errout01Bag_noise = np.array(lin_log_errout01Bag_noise)
    lin_log_errout01Bag_noise = np.mean(lin_log_errout01Bag_noise,axis=0).reshape(-1)

    return lin_errIn01,lin_errInOutDiff01,lin_log_err,lin_log_errout01Bag_noise



def linValidate(W_lin,valDATA,valLabel):

    pred = np.dot(valDATA,W_lin)
    pred[pred >= 0] = 1
    pred[pred < 0] = -1
    err01 = sum(pred != valLabel) / len(valLabel)
    return err01

def logisticGrad(iterNUM, traData, traLabel,valData,valLabel):

    # traData 200,3
    # traLabel 200,1
    # weight 3,1
    W = np.zeros((3,1),dtype=np.float64)
    theta = lambda x: (1 / (1 + np.exp(-x)))

    for _ in range(iterNUM):

        # s = y_n * W^T * x_n
        s = np.dot(traData, W) * traLabel*(-1)
        grad = ( theta(s)* (  (-1)*traLabel * traData))
        grad = np.mean(grad, axis=0).reshape(3, 1)
        W = W - (0.1* grad)

    s = np.dot(valData, W)
    pred = theta(s)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = -1
    res = sum(pred != valLabel) / len(valLabel)
    return res


if __name__ == '__main__':
    for _ in range(10):
        sqrErrIn,\
        linErrOut,\
        lin_log_ErrOut,\
        lin_log_Errout_noise= meanErr(iterNUM=100,noiseTF=1)
        print(f"Q13 : {sqrErrIn}\nQ14 : {linErrOut}\nQ15 : {lin_log_ErrOut}\nQ16 : {lin_log_Errout_noise}")
        print("============================================================================================")





# import matplotlib.pyplot as plt
# Label_ = Label.reshape(-1)
# pos = Data[Label_ == 1][:, 1:]
# neg = Data[Label_ == -1][:, 1:]
# plt.scatter(pos[:, 0:1], pos[:, 1:], s=10, label='Pos')
# plt.scatter(neg[:, 0:1], neg[:, 1:], s=10, label='Neg')
# minX1 = np.min(pos) if np.min(pos) < np.min(neg) else np.min(neg)
# MaxX1 = np.max(pos) if np.max(pos) < np.max(neg) else np.max(neg)
# x_values = [ minX1, MaxX1]
# y_values = [ -(W[0]+W[1]*x_values[0])/W[2],-(W[0]+W[1]*x_values[1])/W[2]]
# plt.plot(x_values, y_values, label='Decision Boundary')
# plt.xlabel('Marks in 1st Exam')
# plt.ylabel('Marks in 2nd Exam')
# plt.legend()
# plt.show()











