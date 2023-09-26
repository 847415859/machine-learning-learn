import math

import  numpy as np
from hmmlearn import  hmm


if __name__ == '__main__':
    # 设定隐藏状态的集合
    states = ["box-1", "box-2", "box-3"]
    n_states = len(states)

    # 设定观察状态的集合
    observations = ["red", "white"]
    n_observations = len(observations)

    # 设定初始状态分布
    start_probability = np.array([0.2, 0.4, 0.4])

    # 设定状态转移概率分布矩阵
    transition_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

    # 设定观测状态概率矩阵
    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])


    # 设置模型参数
    # n_trials = np.array(0,1)
    model = hmm.MultinomialHMM(n_components=n_states,verbose=True)
    # 初始状态概率分布
    model.startprob_ = start_probability
    # 状态转移概率分布矩阵
    model.transmat_ = transition_probability
    # 观测状态概率矩阵
    model.emissionprob_ = emission_probability

    # HMM问题三维特比算法的解码过程，使用和之前一样的观测序列来解码
    seen = np.array([[0, 1, 0]]).T  # 设定观测序列
    model.fit(seen)


    print("After Fit")
    print("startprob \n",model.startprob_)
    print("transmat \n",model.transmat_)
    print("emissionprob \n",model.emissionprob_)

    log_prob, state_seq = model.decode(seen,3)
    score = model.score(seen,3)
    print("score:\n",score, math.exp(score))
    print("球的观测顺序为：\n", ", ".join(map(lambda x: observations[x], seen.flatten())))
    # 注意：需要使用flatten方法，把seen从二维变成一维
    print("最可能的隐藏状态序列为:\n", ", ".join(map(lambda x: states[x], state_seq)))






