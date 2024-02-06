import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """伯努利多臂老虎机"""
    def __init__(self, k):
        # K对应老虎机的数量
        self.probs = np.random.uniform(size=k)  # 随机生成K个概率
        self.best_idx = np.argmax(self.probs)   # 获取概率最大的老虎机的索引
        self.best_prob = self.probs[self.best_idx]  # 获取最大概率
        self.K = k

    def step(self, k):
        # 拉动第k个臂，返回1或0（随机判断是否获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每个臂被拉动的次数
        self.regret = 0.
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        # 计算累积懊悔并保存
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError   # 抽象方法，子类继承方法后若不重写则会报错

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """ ε-Greedy算法 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)     # 随机选择一个臂
        else:
            k = np.argmax(self.estimates)  # 选择当前平均收益最大的臂
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time")
    plt.ylabel("Cumulative Regret")
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


class DecayingEpsilonGreedy(Solver):
    """ 衰减ε-Greedy算法 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1.0 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


if __name__ == "__main__":
    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [
        EpsilonGreedy(BernoulliBandit(10), epsilon=eps) for eps in epsilons
    ]
    epsilon_greedy_solver_names = ["epsilon={}".format(eps) for eps in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
