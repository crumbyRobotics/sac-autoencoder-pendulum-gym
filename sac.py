import gym
import numpy as np
from collections import deque
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
    def __call__(self, inputs, training=False):
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
        mean, stddev = self(states, training)

        # Reparameterizationトリック: 平均0, 標準偏差1のノイズ. サンプリングの代わり
        normal_random = tf.random.normal(mean.shape, mean=0., stddev=1.)
        action_org = mean + stddev * normal_random

        # Squashed Gaussian Policy: actionを-1 ~ 1に変換
        action = tf.tanh(action_org)

        return action, mean, stddev, action_org

    # 学習以外で使う箇所(1アクションを返す)
    def sample_action(self, states, training=False):
        action, mean, _, _ = self.sample_action(
            states.reshape(1, -1), training)
        action = action.numpy()[0]

        # 環境に渡すためにアクションを規格化
        env_action = action * self.action_scale + self.action_centor

        if training:
            return env_action, action
        else:
            # テスト時は平均値を使う
            return mean.numpy()[0] * self.action_scale + self.action_center


class DualQNetwork(keras.Model):
    def __init__(self):
        super().__init__()

        # QNetworkその１
        self.dense1 = keras.layers.Dense(64, activation="relu")
        self.dense2 = keras.layers.Dense(64, activation="relu")
        self.dense3 = keras.layers.Dense(64, activation="relu")
        self.value1 = keras.layers.Dense(1, activation="linear")
        # QNetworkその２
        self.dense4 = keras.layers.Dense(64, activation="relu")
        self.dense5 = keras.layers.Dense(64, activation="relu")
        self.dense6 = keras.layers.Dense(64, activation="relu")
        self.value2 = keras.layers.Dense(1, activation="linear")

        # Optimizer
        self.optimizer = Adam(learning_rate=0.003)

    # Forward pass
    def __call__(self, states, actions, training=False):
        # QNetworkへの入力
        x = tf.concat([states, actions], axis=1)
        # QNetworkその１
        x1 = self.dense1(x)
        x1 = self.dense2(x1)
        x1 = self.dense3(x1)
        q1 = self.value1(x1)
        # QNetworkその２
        x2 = self.dense4(x)
        x2 = self.dense5(x2)
        x2 = self.dense6(x2)
        q2 = self.value2(x2)

        return q1, q2


# mean, stddevの正規分布でactionの確率対数
@tf.function
def compute_logpi(mean, stddev, action):
    a1 = -0.5 * np.log(2*np.pi)
    a2 = -tf.math.log(stddev)
    a3 = -0.5 * (((action - mean) / stddev) ** 2)
    return a1 + a2 + a3

# Squashed Gaussian Policy時のlogπ(a|s)


@tf.function
def compute_logpi_sgp(mean, stddev, action):
    logmu = compute_logpi(mean, stddev, action)


# Pendulum環境を作成
env = gym.make('Pendulum-v0')

# ハイパーパラメータ
buffer_size = 1000  # Experienceのキュー容量
warmup_size = 500  # 学習するかどうかのExperienceの最低限の容量
train_interval = 10  # 学習する制御周期間隔
batch_size = 32  # バッチサイズ
gamma = 0.9  # 割引率
soft_target_tau = 0.02  # Soft TargetでTargetに近づく割合
hard_target_interval = 100  # Hard Targetを更新する間隔

# エントロピーαの目標値: -1xアクション数がいいらしい
target_entropy = -1 * env.action_space.shape[0]

# モデルの定義
policy_model = PolicyModel(env.action_space)
q_model = DualQNetwork()
target_q_model = DualQNetwork()

# NNの初期化: モデルは一度伝搬させないと重みが作成されない仕様
dummy_state = np.random.normal(0, 0.1, size=(1,) + env.observation_space.shape)
dummy_action = np.random.normal(0, 0.1, size=(1,) + env.action_space.shape)
q_model(dummy_state, dummy_action)
target_q_model(dummy_state, dummy_action)
target_q_model.set_weights(q_model.get_weights())

# エントロピーα自動調整用
log_alpha = tf.Variable(0.0, dtype=tf.float32)

# 制御周期ごとに収集する経験は上限を決め、古いものから削除
experiences = deque(maxlen=buffer_size)

all_step_count = 0
all_train_count = 0

# 学習ループ
for episode in range(500):
    state = np.asarray(env.reset())
    done = False
    total_reward = 0
    step = 0

    # １エピソード
    while not done:
        # アクションを決定
        env_action, action = policy_model.sample_action(state, True)

        # step
        n_state, reward, terminated, truncated, _ = env.step(env_action)
        n_state = np.asarray(n_state)
        step += 1
        total_reward += reward
        done = terminated or truncated

        experiences.append({
            "state": state,
            "action": action,
            "reward": reward,
            "n_state": n_state,
            "done": done
        })
        state = n_state

        # train_interval毎に, warmup貯まっていたら学習する
        if len(experience) >= warmup_size and all_step_count % train_interval == 0:
            # モデルの更新
            update_model(
                policy_model,
                q_model,
                target_q_model,
                experiences,
                batch_size,
                gamma,
                log_alpha,
                soft_target_tau,
                hard_target_interval,
                target_entropy,
                all_train_count,
            )
            all_train_count += 1
        all_step_count += 1


# テスト --5回パフォーマンス測定
for episode in range(5):
    state = np.asarray(env.reset())
    env.render()
    done = False
    total_reward = 0
    step = 0

    # １エピソード
    while not done:
        action = policy_model.sample_action(state)
        n_satate, reward, terminated, truncated, _ = env.step(action)
        env.render()
        state = np.asarray(n_state)
        step += 1
        total_reward += reward

    print("{} step, reward: {}".format(step, total_reward))

env.close()
