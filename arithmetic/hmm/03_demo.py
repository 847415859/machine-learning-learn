import numpy as np
from hmmlearn import hmm

states = ["box1", "box2", "box3"]  # 隐藏状态
n_states = len(states)

observations = ["red", "white"]  # 观测状态
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])

emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

model_1 = hmm.MultinomialHMM(n_components=n_states)
model_1.startprob_=start_probability
model_1.transmat_=transition_probability
model_1.emissionprob_=emission_probability
#model_1.n_features = n_observations  # 观测序列的种类
# print(n_observations)

# 使用viterbi进行预测
se = np.array([[0, 1, 0, 1, 1]]).T
print(se)
model_1.fit(se)
logprob, box_index = model_1.decode(se, algorithm='viterbi',lengths=5)
print("颜色:", end="")
print(" ".join(map(lambda t: observations[t], [0, 1, 0,1,1])))
print("盒子:", end="")
print(" ".join(map(lambda t: states[t], box_index)))
print("概率值:", end="")
print(np.exp(logprob)) # 这个是因为在hmmlearn底层将概率进行了对数化，防止出现乘积为0的情况