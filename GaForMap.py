import numpy as np
import random
from time import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# 旅行商问题：从原点出发遍历所有点再回到原点，使距离最短

# 将输入的点存成数组
# 依据数组的下标编码
# 生成距离矩阵，减少后续运算
# 交叉，部分交叉，建立映射表（以便去重复
# 变异，随机交换一个种群的两个点（所以概率提高一点

class GaMap:
    def __init__(self, points, no_bound, DNA_SIZE=None, cross_rate=0.8, mutation=0.1, pop_size=100):
        self.cross_rate = cross_rate  # 杂交率
        self.mutation = mutation  # 变异率
        self.points = np.array(points)  # 点
        self.pop_size = pop_size  # 种群数量

        # 确定DNA大小（碱基数量
        self.DNA_SIZE = 0
        if (DNA_SIZE == None):
            self.DNA_SIZE = len(points)
        else:
            self.DNA_SIZE = DNA_SIZE

        # 距离矩阵
        self.distance = np.zeros((self.DNA_SIZE, self.DNA_SIZE))
        for i in range(self.DNA_SIZE):
            for j in range(self.DNA_SIZE):
                self.distance[i][j] = pow((self.points[i][0] - self.points[j][0]) ** 2 +
                                          (self.points[i][1] - self.points[j][1]) ** 2, 0.5)  # 两点之间的距离

        # 种群
        self.pop = np.zeros((self.pop_size, self.DNA_SIZE - 2), dtype=np.int_)  # 去掉起点和终点,起点终点不可变

        # 编码
        self.code = np.arange(1, self.DNA_SIZE - 1, 1, dtype=np.int_)  # 同上

        # 生成种群
        for i in range(self.pop_size):
            self.pop[i] = self.code.copy()  # 得到code的拷贝
            np.random.shuffle(self.pop[i])  # 洗

    # 解码，好像不需要解码，就这样根据代表点的整数生成适应函数和杂交吧
    def translateDNA(self):
        pass

    # 得到适应度，适应度是距离，越Da越好
    def get_fitness(self, non_negative=False):
        fitness = np.zeros(self.pop_size)
        index = 0

        for people in self.pop:
            for i in range(people.shape[0]):
                if i == 0:
                    fitness[index] += self.distance[0][people[i]]
                else:
                    fitness[index] += self.distance[people[i-1]][people[i]]
            fitness[index] += self.distance[people[-1]][-1]  # 获取最后一个元素到终点的距离
            index += 1

        return 100/fitness
        # return pow(100/fitness, 2)

    # 自然选择
    def select(self):
        fitness = self.get_fitness()  # 说明见GA
        self.pop = self.pop[np.random.choice(np.arange(self.pop.shape[0]), size=self.pop.shape[0], replace=True,
                                             p=fitness / np.sum(fitness))]

    # 染色体交叉
    def crossover(self):
        for people in self.pop:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.pop.shape[0], size=1)  # 产生0~pop的行数之间的一个随机数，即要进行交叉的染色体编号
                cross_area = random.sample(range(people.shape[0]), 2)  # 生成两个要交换的数

                if cross_area[0] > cross_area[1]:  # 保证第一个数小于第二个数
                    cross_area[0], cross_area[1] = cross_area[1], cross_area[0]

                temp = self.pop[i_][0]  # 这样的得到的数组还有一个中括号，可能因为是引用
                crosser = np.array(temp[cross_area[0]:cross_area[1]].copy())
                becrosser = people[cross_area[0]:cross_area[1]].copy()

                for i in range(crosser.shape[0]):
                    for j in range(crosser.shape[0]):  # 将交换区内的重复点归零
                        if (becrosser[j] != 0) & (becrosser[j] == crosser[i]):
                            becrosser[j] = 0
                            crosser[i] = 0
                            break

                people[cross_area[0]:cross_area[1]] = self.pop[i_][0][cross_area[0]:cross_area[1]].copy()  # 单方面杂交
                for i in range(crosser.shape[0]):  # 先替换两个交换区之间的不重复元素（时间复杂度 n^2 / 4
                    if crosser[i] != 0:
                        for j in range(cross_area[0]):
                            if crosser[i] == people[j]:
                                if becrosser[i] != 0:
                                    people[j] = becrosser[i]
                                    becrosser[i] = 0
                                    crosser[i] = 0
                                    break
                        if crosser[i] != 0:
                            for j in range(cross_area[1], people.shape[0]):
                                if crosser[i] == people[j]:
                                    if becrosser[i] != 0:
                                        people[j] = becrosser[i]
                                        becrosser[i] = 0
                                        crosser[i] = 0
                                        break
                i = 0
                while i < crosser.shape[0]:  # 删除为0的元素减少遍历时间
                    if crosser.size == 0:
                        break
                    if crosser[i] == 0:
                        crosser = np.delete(crosser, i)
                        i -= 1
                    i += 1
                i = 0
                while i < becrosser.shape[0]:
                    if becrosser.size == 0:
                        break
                    if becrosser[i] == 0:
                        becrosser = np.delete(becrosser, i)
                        i -= 1
                    i += 1

                for i in range(crosser.shape[0]):  # 再替换两个交换区之间的重复元素
                    for j in range(cross_area[0]):
                        if crosser[i] == people[j]:
                            people[j] = becrosser[i]
                            crosser[i] = 0
                            break
                    if crosser[i] != 0:
                        for j in range(cross_area[1], people.shape[0]):
                            if crosser[i] == people[j]:
                                people[j] = becrosser[i]
                                crosser[i] = 0
                                break
                    if crosser[i] != 0:
                        print("bug hakken!")
                        exit(0)

    # 基因变异
    def mutate(self):
        for people in self.pop:
            if np.random.rand() < self.mutation:
                change = random.sample(range(people.shape[0]), 2)  # 生成两个要交换的数
                people[change[0]], people[change[1]] = people[change[1]], people[change[0]]  # 交换
                # temp = people[change[0]]
                # np.delete(people, change[0], 0)
                # np.insert(people, change[1], temp, 0)

    # 进化
    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()

    # 重置
    def reset(self):
        pass

    # 打印当前状态日志
    def log(self):
        pass

    # 开始
    def start(self, iter_time=200):
        percent = 5
        for _ in range(iter_time):
            if (_ / iter_time) >= (percent/100):
                print("#", end="")
                percent += 5
            self.evolution()
        print()

    # 作图
    def plot(self):
        fitness = self.get_fitness()
        maxindex = np.argmax(fitness)
        print(self.pop[maxindex])
        print(fitness)
        print(100/fitness[maxindex])
        plot_x = []
        plot_y = []

        plot_x.append(self.points[0][0])  # 添加起点
        plot_y.append(self.points[0][1])
        for i in self.pop[maxindex]:
            plot_x.append(self.points[i][0])  # 添加途经点
            plot_y.append(self.points[i][1])
        plot_x.append(self.points[-1][0])  # 添加终点
        plot_y.append(self.points[-1][1])

        plt.figure(figsize=(8, 4))  # 里面的参数定义了分辨率
        plt.plot(plot_x, plot_y, color="#FF6666", marker='o', alpha=0.8, lw=2, label='x')
        plt.legend()
        plt.show()


# 旅行者路径：起点和终点要一样
point = np.array([[1, 1], [1, 7], [5, 2], [6, 4], [2, 3], [7, 4], [5, 1], [6, 2], [8, 9], [4, 7], [1, 1]])
begin = time()
ga = GaMap(point, 0)
ga.start()
print("算法运行时间:", time()-begin)
ga.plot()

