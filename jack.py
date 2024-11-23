import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import os
import seaborn as sns

GAMMA = 0.9
MAX_CAR_NUM = 20
E1out = 3
# 地点1租车的期望
E1in = 3
# 地点1还车的期望
E2out = 4
E2in = 2

ONE_CAR_INPUT = 10

ACTION = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
ACTION_NUM = 11

terminal = 1

E_greedy = 0
E_greedy_M = E_greedy / MAX_CAR_NUM

P1 = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
# P1[x][y]对应了地点1中由x辆车变成y辆车的概率
P2 = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
# R = np.zeros(shape=(MAX_CAR_NUM+1, MAX_CAR_NUM+1)).astype("float32")
# # R[x][y]表示地点1和地点2分别有x和y辆车时获得奖励的期望
# 不再使用，每个状态的收获是动态的，而不是简单的期望
Rs = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")


# R[x][y]表示Rt为状态为(x,y)时的Rs，表示的是移动车辆以后的状态


def init_p():
    for i in range(0, MAX_CAR_NUM + 1):
        for d_in in range(0, MAX_CAR_NUM + 1):
            for d_out in range(0, MAX_CAR_NUM + 1):
                j_to = max(i - d_out, 0)
                j_to = min(j_to + d_in, MAX_CAR_NUM)
                P1[i][j_to] += poisson.pmf(k=d_out, mu=E1out) * poisson.pmf(k=d_in, mu=E1in)
                P2[i][j_to] += poisson.pmf(k=d_out, mu=E2out) * poisson.pmf(k=d_in, mu=E2in)

            # if i < E1out:
            #     R[i][j] += i
            # else:
            #     R[i][j] += E1out
            # if j < E2out:
            #     R[i][j] += j
            # else:
            #     R[i][j] += E2out


def init_rs():
    # i,j表示执行动作以后状态两地分别有i,j辆车
    # Rs是定值
    for i in range(0, MAX_CAR_NUM + 1):
        for j in range(0, MAX_CAR_NUM + 1):
            for e1out in range(0, MAX_CAR_NUM + 1):
                for e2out in range(0, MAX_CAR_NUM + 1):
                    reward = float((min(i, e1out) + min(j, e2out)) * ONE_CAR_INPUT)
                    # 当前有四辆，借了五辆，用借五辆的概率，借走四辆的收获
                    Rs[i][j] += poisson.pmf(k=e1out, mu=E1out) * poisson.pmf(k=e2out, mu=E2out) * reward


def get_state_action_v(state, action, value):
    i = state[0]
    j = state[1]
    now_i = i + action
    now_j = j - action
    # now_i,now_j表示执行动作以后状态两地分别有i,j辆车
    results = 0
    if 0 <= now_i <= MAX_CAR_NUM and 0 <= now_j <= MAX_CAR_NUM:
        for to_i in range(0, MAX_CAR_NUM + 1):
            for to_j in range(0, MAX_CAR_NUM + 1):
                results += P1[now_i][to_i] * P2[now_j][to_j] * value[to_i][to_j] * GAMMA
        results += Rs[now_i][now_j] - abs(action) * 2  # 减去移动车辆的代价
    else:
        results = 0
    return results


def update_state_action_v(value):
    state_action_value = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1, ACTION_NUM)).astype("float32")
    for i in range(0, MAX_CAR_NUM + 1):
        for j in range(0, MAX_CAR_NUM + 1):
            # 遍历所有状态
            for which_action in range(ACTION_NUM):
                # 遍历所有action,并找出最大值
                now_action = ACTION[which_action]
                now_i = i + now_action
                now_j = j - now_action
                # now_i,now_j表示执行动作以后两地分别有now_i,now_j辆车
                if 0 <= now_i <= MAX_CAR_NUM and 0 <= now_j <= MAX_CAR_NUM:
                    for to_i in range(0, MAX_CAR_NUM + 1):
                        for to_j in range(0, MAX_CAR_NUM + 1):
                            state_action_value[i][j][which_action] += P1[now_i][to_i] * P2[now_j][to_j] * value[to_i][
                                to_j] * GAMMA
                    state_action_value[i][j][which_action] += Rs[now_i][now_j] - abs(now_action) * 2  # 减去移动车辆的代价
                else:
                    state_action_value[i][j][which_action] = 0
    return state_action_value


def get_E_greedy_strategy(state_action_value):
    strategy = np.ones(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1, ACTION_NUM)).astype("float32")
    strategy = strategy * E_greedy_M
    for i in range(0, MAX_CAR_NUM + 1):
        for j in range(0, MAX_CAR_NUM + 1):
            values = state_action_value[i][j]
            max_action_index = np.argmax(values)
            strategy[i][j][max_action_index] += 1 - E_greedy
    return strategy


def get_greedy_action(state_action_value):
    best_action = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("int")
    for i in range(0, MAX_CAR_NUM + 1):
        for j in range(0, MAX_CAR_NUM + 1):
            values = state_action_value[i][j]
            best_action[i][j] = ACTION[np.argmax(values)]
    return best_action


def SYN_VALUE_ITER():
    # 同步

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.flatten()

    value = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
    temp_V = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
    iterations = 0
    while True:
        state_action_value = update_state_action_v(value)
        for i in range(0, MAX_CAR_NUM + 1):
            for j in range(0, MAX_CAR_NUM + 1):
                values = state_action_value[i][j]
                temp_V[i][j] = np.max(values)
        errand = abs(temp_V - value).max()
        print("now's iterations: {} , errand is: {}\n".format(iterations, errand))
        iterations += 1
        if errand < terminal:
            strategy = get_greedy_action(state_action_value)
            break
        value = temp_V.copy()  # 把新表值赋给旧表
        # 画图
    fig = sns.heatmap(np.flipud(strategy), cmap="rainbow", ax=axes[0])
    fig.set_ylabel('# cars at first location', fontsize=10)
    fig.set_yticks(list(reversed(range(MAX_CAR_NUM + 1))))
    fig.set_xlabel('# cars at second location', fontsize=10)
    fig.set_title('policy', fontsize=10)
    fig = sns.heatmap(np.flipud(value), cmap="rainbow", ax=axes[1])
    fig.set_ylabel('# cars at first location', fontsize=10)
    fig.set_yticks(list(reversed(range(MAX_CAR_NUM + 1))))
    fig.set_xlabel('# cars at second location', fontsize=10)
    fig.set_title('value', fontsize=10)
    plt.show()
    return value, strategy


def ASY_VALUE_ITER():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.flatten()

    # 异步
    value = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
    iterations = 0
    while True:
        old_value = value.copy()
        for i in range(0, MAX_CAR_NUM + 1):
            for j in range(0, MAX_CAR_NUM + 1):
                max_value = 0
                for which_action in range(ACTION_NUM):
                    # 遍历所有action,并找出最大值
                    now_action = ACTION[which_action]
                    this_v = get_state_action_v((i, j), now_action, value)
                    if this_v > max_value:
                        max_value = this_v
                value[i][j] = max_value
        errand = abs(old_value - value).max()
        print("now's iterations: {} , errand is: {}\n".format(iterations, errand))
        iterations += 1
        if errand < terminal:
            state_action_value = update_state_action_v(value)
            strategy = get_greedy_action(state_action_value)
            break
            # 画图
    fig = sns.heatmap(np.flipud(strategy), cmap="rainbow", ax=axes[0])
    fig.set_ylabel('# cars at first location', fontsize=10)
    fig.set_yticks(list(reversed(range(MAX_CAR_NUM + 1))))
    fig.set_xlabel('# cars at second location', fontsize=10)
    fig.set_title('policy', fontsize=10)
    fig = sns.heatmap(np.flipud(value), cmap="rainbow", ax=axes[1])
    fig.set_ylabel('# cars at first location', fontsize=10)
    fig.set_yticks(list(reversed(range(MAX_CAR_NUM + 1))))
    fig.set_xlabel('# cars at second location', fontsize=10)
    fig.set_title('value', fontsize=10)
    plt.show()
    return value, strategy


def POLICY_ITER():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.flatten()

    iterations_num = 5
    value = np.ones(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
    best_strategy = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
    for iterations in range(iterations_num):
        last_value = value.copy()

        state_action_value = update_state_action_v(value)
        strategy = get_E_greedy_strategy(state_action_value)

        while True:
            old_value = value.copy()
            state_action_value = update_state_action_v(value)
            value = np.zeros(shape=(MAX_CAR_NUM + 1, MAX_CAR_NUM + 1)).astype("float32")
            for i in range(0, MAX_CAR_NUM + 1):
                for j in range(0, MAX_CAR_NUM + 1):
                    # 遍历所有状态
                    for which_action in range(ACTION_NUM):
                        # 遍历所有action
                        value[i][j] += strategy[i][j][which_action] * state_action_value[i][j][which_action]
            errand = abs(old_value - value).max()
            print("strategy evaluate: errand is: {}\n".format(errand))
            if errand < terminal:
                break

        errand = abs(last_value - value).max()
        print("now iterations is:{} errand is: {}\n".format(iterations, errand))
        best_strategy = get_greedy_action(state_action_value)
        fig = sns.heatmap(np.flipud(best_strategy), cmap="rainbow", ax=axes[iterations])
        # 定义标签与标题
        fig.set_ylabel('# cars at first location', fontsize=10)
        fig.set_yticks(list(reversed(range(MAX_CAR_NUM + 1))))
        fig.set_xlabel('# cars at second location', fontsize=10)
        fig.set_title('policy {}'.format(iterations), fontsize=10)
    fig = sns.heatmap(np.flipud(value), cmap="rainbow", ax=axes[5])
    fig.set_ylabel('# cars at first location', fontsize=10)
    fig.set_yticks(list(reversed(range(MAX_CAR_NUM + 1))))
    fig.set_xlabel('# cars at second location', fontsize=10)
    fig.set_title('value', fontsize=10)
    plt.show()
    return value, best_strategy


def show(in_array):
    plt.imshow(in_array, cmap='plasma', vmin=np.min(in_array), vmax=np.max(in_array))  # 使用 'plasma' 渐变色
    plt.colorbar()  # 显示颜色条
    plt.title('Float32 2D Array Visualization (Darker for Larger Values with Plasma)')
    plt.show()


if_load = True
# 是否从保存下来的文件中加载P1,P2,Rs数组，若不加载，则重新生成并保存
if if_load:
    P1 = np.load('P1.npy')
    P2 = np.load('P2.npy')
    Rs = np.load('Rs.npy')
    # show(np.flipud(P1))
    # show(np.flipud(P2))
    # show(np.flipud(Rs))
else:
    os.chdir(r'C:\Users\lenovo\Desktop\RLGo')
    init_p()
    show(np.flipud(P1))
    show(np.flipud(P2))
    init_rs()
    show(np.flipud(Rs))
    np.save('P1.npy', P1)
    np.save('p2.npy', P2)
    np.save('Rs.npy', Rs)

# Value, Strategy = ASY_VALUE_ITER()
Value, Strategy = SYN_VALUE_ITER()
# Value, Strategy = POLICY_ITER()
# show(np.flipud(Strategy))
# show(np.flipud(Value))
