import gym

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


class PolicyModel(keras.Model):
    def __init__(self, action_space):
        super().__init__()

        self.action_space = action_space

        # Envアクション用
        self.action_center = (action_space.high + action_space.low) / 2
        self.action_scale = action_space.high - self.action_center

        # DNNの各層を定義
        self.dense1 = keras.layers.Dense(64, activation="relu")
        self.dense2 = keras.layers.Dense(64, activation="relu")
        self.dense3 = keras.layers.Dense(64, activation="relu")
        self.pi_mean = keras.layers.Dense(
            action_space.shape[0], activation="linear")
        self.pi_stddev = keras.layers.Dense(
            action_space.shape[0], activation="linear")

        # Optimizer
        self.optimizer = Adam(learning_rate=0.003)

    # DNNのForward pass
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        mean = self.pi_mean(x)
        stddev = self.pi_stddev(x)

        # σ > 0になるように指数関数で変換
        stddev = tf.exp(stddev)

        return mean, stddev

    # 学習（自動勾配内）でも使う関数
    @tf.function
    def sample_action(self, states, training=False):
        mean, stddev

        # CartPole環境を作成
env = gym.make('CartPole-v1')


# Q学習用のDNNを作成
obs_shape = env.observation_space.shape  # 環境の状態の形式(shape)

nb_actions = env.action_space.n  # 環境の取りうるアクション数

lr = 0.001  # 学習率

c = input_ = keras.layers.Input(shape=obs_shape)
c = keras.layers.Dense(10, activation="relu")(c)
c = keras.layers.Dense(10, activation="relu")(c)
c = keras.layers.Dense(nb_actions, activation="linear")(c)
model = keras.Model(input_, c)
model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=["mae"])
model.summary()  # 4 (state数) -> 10 -> 10 -> 2 (action数) の４層DNN


# シミュレーションしてみる
state = env.reset(seed=42)
frame = env.render()

total_reward = 0
step = 0

frames = [frame]
while True:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1

    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        break

print("step: {}, reward: {}".format(step, total_reward))

env.close()
