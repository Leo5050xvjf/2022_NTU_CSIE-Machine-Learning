


import numpy as np
import random

def PLA(x0 = 1,scale = 1,normalize = 0):
    # 將data 進行初始化動作，包含x0設定 、 x 的scale 、x 的normalize
    data_bag = []
    with open("./hw1_train.dat.txt") as file:
        for data in file:
            data= [float(d) for d in data.split()]
            data.insert(0,x0)
            if scale > 1:
                for _ in range(11):
                    data[_] = data[_] * scale
            if normalize == 1:
                d = np.array(data[:11])
                norm = np.sqrt(d.dot(d))
                for _ in range(11):
                    data[_] = data[_] /norm
            data_bag.append(data)

    # 以下進行1000次的PLA
    w_pla_bag = []
    for _ in range(1000):
        w_pla = [0 for i in range(11)]
        w_pla = np.array(w_pla)
        counter  = 0
        # 需要從0-99中亂數選出500個數字
        random_seed = np.random.randint(0,100,500)
        # 脫離while迴圈的條件為，連續抽出500筆x，並且利用w_pla_i判別正確，則將此次的w_pla_i記錄在w_pla_bag
        while(counter < 500 ):
            # random_seed 為 0-99的亂數列表 共500個
            for seed in random_seed:
                # 找到第一個隨機點
                point = data_bag[seed]
                # 判別此點是否分類正確
                x = np.array(point[:11])
                label = point[-1]
                dotValue = w_pla.dot(x)
                # (若是內積w_pla 和 point_x 與 label 不同號) or (內積為0)， 則判定此點為錯誤點，因此要更新，並將counter歸0重新計算。
                if dotValue * label <=  0:
                    w_pla = w_pla + (label * x)
                    counter = 0
                    break
                # 若兩者同號則  dotValue * label > 0 ，此時不做更新
                counter += 1

        w_pla_bag.append(w_pla.dot(w_pla))
    # 將1000次的結果取平均並回傳
    w_pla_average = sum(w_pla_bag) / len(w_pla_bag)
    return w_pla_average
if __name__ == "__main__":
    # Q13 x0 = 1 , no scale  ,no normalize
    ans = PLA(1,1,0)
    print(f"the Q13 w_pla is {ans}")
    # Q14 x0 = 1 , scale = 2 ,no normalize
    ans = PLA(1,2,0)
    print(f"the Q14 w_pla is {ans}")

    # Q15 x0 = 1 , no scale  ,normalize
    ans = PLA(1,1,1)
    print(f"the Q15 w_pla is {ans}")

    # Q16 x0 = 0 , no scale  ,no normalize
    ans = PLA(0,1,0)
    print(f"the Q16 w_pla is {ans}")


