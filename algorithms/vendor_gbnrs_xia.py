"""VERBATIM copy of the functions in GBNRS_new/xin_de.py from the authors'
public repository https://github.com/syxiaa/GBNRS (archive GBNRS_new.rar),
Xia et al., "GBNRS: A Novel Rough Set Algorithm for Fast Adaptive Attribute
Reduction in Classification", IEEE TKDE 34(3), 2022.
Copied 2026-11-18. Only the
`mean_std` helper and the `__main__` experiment block (hard-coded Windows paths)
were removed; nothing else was changed. Do not edit: wrap it in GBNRS_Xia.py.
Source SHA-256 and commit: see BASELINES.md.
"""
import copy
import csv
import time
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import warnings
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

warnings.filterwarnings("ignore")


# Calculate the label and purity of GB
def calLabelPurity(data):
    if len(data) > 1:
        count = Counter(data[:, -1])
        label = max(count, key=count.get)
        purity = count[label] / len(data)
    else:
        label = data[0][-1]
        purity = 1.0
    return label, purity


# Calculate the center and radius of GB
def calCenterRadius(data):
    center = np.mean(data, axis=0)
    dis = calculateDist(data, center)
    radius = np.mean(dis)
    return center, radius


# Calculate the Euclidean distance between objects
def calculateDist(A, B, flag=0):
    if (flag == 0):
        return np.sqrt(np.sum((A - B) ** 2, axis=1))
    else:
        return np.sqrt(np.sum((A - B) ** 2))


def generateOmegaCluster(data, attribute):
    cluster = []
    todo_data = []
    count = Counter(data[:, -1])
    label = max(count, key=count.get)
    Omega_Group = data[data[:, -1] == label]
    Omega_Group = Omega_Group[:, attribute]
    center = np.mean(Omega_Group, axis=0)
    dis = np.around(calculateDist(data[:, attribute], center), 6)
    Omega_Group_dis = dis[data[:, -1] == label]
    radius = np.around(np.mean(Omega_Group_dis), 6)
    for i in range(len(data)):
        if dis[i] <= radius:
            cluster.append(data[i])
        else:
            todo_data.append(data[i])
    cluster = np.array(cluster)
    todo_data = np.array(todo_data)
    return cluster, todo_data


def removeConflicts(ball_list):
    gb_remove_tmp = []
    for i in range(len(ball_list) - 1):
        if ball_list[i] in gb_remove_tmp:
            continue
        for j in range(i + 1, len(ball_list)):
            if (ball_list[j] in gb_remove_tmp):
                continue
            if (ball_list[i].label != ball_list[j].label) and (calculateDist(
                    ball_list[i].center, ball_list[j].center,
                    flag=1) <= abs(ball_list[i].radius - ball_list[j].radius)):
                ball_list[j].data = np.concatenate(
                    (ball_list[j].data, ball_list[i].data))
                ball_list[j].label, ball_list[j].purity = calLabelPurity(
                    ball_list[j].data)
                ball_list[j].center, ball_list[j].radius = calCenterRadius(
                    ball_list[j].data_no_label)
                gb_remove_tmp.append(ball_list[i])
                break
    for ball in set(gb_remove_tmp):
        ball_list.remove(ball)
    return ball_list


def splitGB(gb_data, attribute):
    label_num = len(np.unique(gb_data[:, -1]))
    gb_data_num = len(gb_data)
    ball_list = []
    ball_list_new = []
    while True:#label_num != gb_data_num
        cluster, gb_data_tmp = generateOmegaCluster(gb_data, attribute)
        # print("len(cluster)", len(cluster))
        if len(cluster) >= 1:  # 去掉离群点构成粒球
            new_ball = GranularBall(cluster, attribute)
            ball_list.append(new_ball)
        else:#不可再分的数据点作为一个球
            new_ball = GranularBall(gb_data_tmp, attribute)
            ball_list.append(new_ball)
            break
        gb_data = gb_data_tmp
        gb_data_num = len(gb_data)
        if gb_data_num == 1:
            new_ball1 = GranularBall(gb_data, attribute)
            ball_list.append(new_ball1)
            break
        elif gb_data_num == 0:
            break
            # label_num = len(np.unique(gb_data[:, -1]))
        #
        # else:
        #     break
    if len(ball_list) > 1:  # Eliminate conflicts消除冲突
        ball_list_new = removeConflicts(ball_list)
    else:
        ball_list_new = ball_list
    return ball_list_new


class GranularBall:
    """class of the granular ball"""

    def __init__(self, data, attribute):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.data = data[:, :]
        self.attribute = attribute
        self.data_no_label = data[:, attribute]
        self.num, self.dim = self.data_no_label.shape
        self.center, self.radius = calCenterRadius(self.data_no_label)
        self.label, self.purity = calLabelPurity(data)

    # 删除
    def split_2balls(self):
        Balls = []
        return Balls


def funtion(ball_list, minsam, attribute):
    Ball_list = ball_list
    Ball_list = sorted(Ball_list, key=lambda x: -x.radius, reverse=True)
    ballsNum = len(Ball_list)
    j = 0
    ball = []
    while True:
        if len(ball) == 0:
            ball.append([Ball_list[j].center, Ball_list[j].radius, Ball_list[j].label, Ball_list[j].num])
            j += 1
        else:
            # flag = False
            for index, values in enumerate(ball):
                if values[2] != Ball_list[j].label and (
                        np.sum((values[0] - Ball_list[j].center) ** 2) ** 0.5) < (
                        values[1] + Ball_list[j].radius) and Ball_list[j].radius > 0 and Ball_list[j].num >= minsam / 2 and \
                        values[3] >= minsam / 2:
                    # if two ball's label is different and overlapped，in this step,we can continut split positive domain can keep boundary region don't change, but this measure is unnecessary
                    balls = splitGB(Ball_list[j].data, attribute)
                    if len(balls) == 0:
                        print(balls)
                        Ball_list.pop(j)
                        ballsNum -= 1
                        continue
                    if len(balls) > 1:
                        Ball_list[j] = balls[0]
                        Ball_list.extend(balls[1:])
                        ballsNum += len(balls) - 1
                    else:
                        Ball_list[j] = balls[0]

            # if flag == False:
            #     # print(8)
            ball.append([Ball_list[j].center, Ball_list[j].radius, Ball_list[j].label, Ball_list[j].num])
            j += 1
        if j >= ballsNum:
            break
    return Ball_list


def overlap_resolve(ball_list, data, attributes_reduction, min_sam):
    Ball_list = funtion(ball_list, min_sam, attributes_reduction)  # conitinue to split ball which are overlapped
    # # do last overlap for granular ball aimed raise ball's quality
    # while True:
    #     init_center = []  # ball's center
    #     Ball_num1 = len(Ball_list)
    #     for i in range(len(Ball_list)):
    #         init_center.append(Ball_list[i].center)
    #     ClusterLists = KMeans(init=np.array(init_center), n_clusters=len(Ball_list)).fit(data[:, attributes_reduction])
    #     data_label = ClusterLists.labels_
    #     ball_list = []
    #     for i in set(data_label):
    #         ball_list.append(GranularBall(data[data_label == i, :], attributes_reduction))
    #     Ball_list = funtion(ball_list, min_sam, attributes_reduction) # conitinue to split ball which are overl
    #     Ball_num2 = len(Ball_list)  # get ball numbers
    #     if Ball_num1 == Ball_num2:  # stop until ball's numbers don't change
    #         break
    return Ball_list


class GBList:
    """class of the list of granular ball"""

    def __init__(self, data=None, attribu=[]):
        self.data = data[:, :]
        self.attribu = attribu
        self.granular_balls = [GranularBall(self.data, self.attribu)]  # gbs is initialized with all data用所有数据初始化GBS

    def init_granular_balls(self, purity=0.996, min_sample=1):
        """
        Split the balls, initialize the balls list.
        purty=1,min_sample=2d
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        GB_List = self.granular_balls
        i = 0  # cursor
        GB_num = len(GB_List)
        j = 1
        while True:
            # print("\n循环次数", j)
            j += 1
            # print("if前len(GB_List)", len(GB_List))
            if GB_List[i].purity < purity and GB_List[i].num > min_sample:#and GB_List[i].num > min_sample
                new_split_gbs = splitGB(GB_List[i].data, self.attribu)
                # print("len(new_split_gbs)", len(new_split_gbs))
                # print("长度", len(new_split_gbs[0].data))
                # print("长度", len(GB_List[i].data))
                if len(new_split_gbs) > 1:
                    GB_List[i] = new_split_gbs[0]
                    GB_List.extend(new_split_gbs[1:])
                elif len(new_split_gbs) == 1:
                    i+=1
            else:
                i += 1
            GB_num = len(GB_List)
            # print("if后len(GB_List)", len(GB_List))
            if i == GB_num:
                break
        # print("len(GB_List)", len(GB_List))
        if len(GB_List) == 0:
            return
        ball_lists = GB_List
        Bal_List = overlap_resolve(ball_lists, self.data, self.attribu, min_sample)  # do overlap
        self.granular_balls = Bal_List
        self.get_data()
        self.data = self.get_data()
        return

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_r(self):
        """
        :return: return radius r
        """
        return np.array(list(map(lambda x: x.radius, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)

    def del_ball(self, purty=0., num_data=0):
        # delete ball
        T_ball = []
        for ball in self.granular_balls:
            if ball.purity >= purty and ball.num >= num_data:
                T_ball.append(ball)
        self.granular_balls = T_ball.copy()
        self.data = self.get_data()

    def R_get_center(self, i):
        # get ball's center
        attribu = self.attribu
        attribu.append(i)
        centers = []
        for ball in range(self.granular_balls):
            center = []
            data_no_label = ball.data[:, attribu]
            center = data_no_label.mean(0)
            centers.append(center)
        return centers


def attribute_reduce(data, pur=1, d2=2):
    """
    :param data: dataset[属性，标签，索引]
    :param pur: purity threshold
    :param d2: min_samples, the default value is 2
    :return: reduction attribute
    """

    re_attribu = [i for i in range(len(data[0]) - 1)]  ##属性的长度
    attribu = re_attribu.copy()

    # print('N_attribute1', N_attribu)
    gb = GBList(data, re_attribu)  # create the list of granular balls###修改产生粒球的方法
    gb.init_granular_balls(purity = pur, min_sample = d2)  # initialize the list初始化列表
    ball_list1 = gb.granular_balls
    # for bal in ball_list1:
    # 	if bal.purity<=0.999:
    # 		print(bal.center,bal.r)
    # 	else:
    # 		print(bal.center,bal.r,bal.label)
    Pos_num = 0
    for ball in ball_list1:
        if ball.purity == 1:
            Pos_num += ball.num  # find the current  domain samples找到当前的域示例
    bal_num = Pos_num   ####先赋值全部属性的正域样本数量

    while len(re_attribu):
        N_i = -1
        N_bal_num = -9999
        for i in re_attribu:
            # print("re_attribu1", re_attribu)
            N_attribu = copy.deepcopy(attribu)  # 初始是全集
            N_attribu.remove(i)
            gb = GBList(data, N_attribu)  # create the list of granular balls###修改产生粒球的方法
            gb.init_granular_balls(purity=pur, min_sample=d2)  # initialize the list初始化列表
            ball_list1 = gb.granular_balls
            # for bal in ball_list1:
            # 	if bal.purity<=0.999:
            # 		print(bal.center,bal.r)
            # 	else:
            # 		print(bal.center,bal.r,bal.label)
            Pos_num = 0
            for ball in ball_list1:
                if ball.purity == 1:
                    Pos_num += ball.num  # find the current  domain samples找到当前的域示例
            if Pos_num >= N_bal_num:
                N_bal_num = Pos_num
                N_i = i
            print("min_Pos_num_i", N_attribu, N_bal_num, N_i)
        print("delete", bal_num, N_bal_num)
        if N_bal_num >= bal_num:##添加5%的浮动* 0.96
            bal_num = N_bal_num
            attribu.remove(N_i)
            re_attribu.remove(N_i)
        else:
            return attribu
    return attribu
