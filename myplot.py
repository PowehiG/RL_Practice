import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 模拟数据
# 你可以将这一部分替换为实际的数据加载
def simulate_data(steps=1000000, envs=7, algos=6):
    import numpy as np
    np.random.seed(42)
    steps_data = np.linspace(0, steps, num=500)
    return {
        f'Env{env}': pd.DataFrame({
            'steps': steps_data,
            'Algorithm1': np.random.rand(500) * 1000 + env * 100,
            'Algorithm2': np.random.rand(500) * 1000 + env * 100,
            'Algorithm3': np.random.rand(500) * 1000 + env * 100,
            'Algorithm4': np.random.rand(500) * 1000 + env * 100,
            'Algorithm5': np.random.rand(500) * 1000 + env * 100,
            'Algorithm6': np.random.rand(500) * 1000 + env * 100,
        }) for env in range(envs)
    }

# 加载或模拟每个环境的数据
data = simulate_data()

# 定义环境和算法
environments = ['HalfCheetah-v1', 'Hopper-v1', 'InvertedDoublePendulum-v1',
                'InvertedPendulum-v1', 'Reacher-v1', 'Swimmer-v1', 'Walker2d-v1']
algorithms = ['Algorithm1', 'Algorithm2', 'Algorithm3', 'Algorithm4', 'Algorithm5', 'Algorithm6']

# 创建一个子图的画布
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 7))

# 绘制每个环境的图表
for idx, (env, ax) in enumerate(zip(environments, axes.flat)):
    df = data[f'Env{idx}']
    for algo in algorithms:
        sns.lineplot(x='steps', y=algo, data=df, ax=ax, label=algo)
    ax.set_title(env)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Score')
    ax.legend(loc='best')

plt.tight_layout()
plt.show()