import random
from collections import deque
from cmath import isnan

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from matplotlib import pyplot as plt

import gym

import numpy as np


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
        self.pi_mean = keras.layers.Dense(action_space.shape[0], activation="linear")
        self.pi_stddev = keras.layers.Dense(action_space.shape[0], activation="linear")

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
    def sample_actions(self, states, training=False):
        mean, stddev = self(states, training)

        # Reparameterizationトリック: 平均0, 標準偏差1のノイズ. サンプリングの代わり
        normal_random = tf.random.normal(mean.shape, mean=0., stddev=1.)
        action_org = mean + stddev * normal_random

        # Squashed Gaussian Policy: actionを-1 ~ 1に変換
        action = tf.tanh(action_org)

        return action, mean, stddev, action_org

    # 学習以外で使う箇所(1アクションを返す)
    def sample_action(self, states, training=False):
        action, mean, stddev, action_org = self.sample_actions(states.reshape(1, -1), training)
        # print("sample_action:", action.numpy(), mean.numpy(), stddev.numpy(), action_org.numpy())
        action = action.numpy()[0]

        # 環境に渡すためにアクションを規格化
        env_action = action * self.action_scale + self.action_center

        if training:
            return env_action, action
        else:
            # テスト時は平均値を使う
            return tf.tanh(mean.numpy()[0]) * self.action_scale + self.action_center


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
    def call(self, states, actions, training=False):
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


# tanhで変換されたactionのlogπ(a|s)を計算
@tf.function
def compute_logpi_sgp(mean, stddev, action):
    logmu = compute_logpi(mean, stddev, action)
    tmp = 1 - tf.tanh(action) ** 2
    tmp = tf.clip_by_value(tmp, 1e-10, 1.0)  # log(0)回避
    logpi = logmu - tf.reduce_sum(tf.math.log(tmp), axis=1, keepdims=True)
    return logpi


def update_model(
        policy_model,
        q_model,  # 重みを勾配法で更新するQネットワーク
        target_q_model,  # q_modelへ緩やかに重みを変化させるQネットワーク
        experiences,
        batch_size,
        gamma,
        log_alpha,
        soft_target_tau,
        hard_target_interval,
        target_entropy,
        all_train_count):

    # 方策エントロピーの反映率αを計算
    alpha = tf.math.exp(log_alpha)

    # ランダムに経験を取得してバッチ作成
    batchs = random.sample(experiences, batch_size)

    # 経験データ整形
    states = np.asarray([e["state"] for e in batchs])
    n_states = np.asarray([e["n_state"] for e in batchs])
    actions = np.asarray([e["action"] for e in batchs])
    rewards = np.asarray([e["reward"] for e in batchs]).reshape((-1, 1))
    dones = np.asarray([e["done"] for e in batchs]).reshape((-1, 1))

    # Q(n_states, n_actions)を計算
    # ポリシーより次の状態のアクションを取得
    n_actions, n_means, n_stddevs, n_action_orgs = policy_model.sample_actions(n_states)
    # 次の状態のアクションのlogpiを計算
    n_logpi = compute_logpi_sgp(n_means, n_stddevs, n_action_orgs)
    # 2つのQ値(targetの方)から小さい方を採用(Clipped Double Q Learning)
    n_q1, n_q2 = target_q_model(n_states, n_actions)

    # q_valsに出力が近づくように重みを更新
    q_vals = rewards + (1 - dones) * gamma * tf.minimum(n_q1, n_q2) - (alpha * n_logpi)

    with tf.GradientTape() as tape:
        q1, q2 = q_model(states, actions, training=True)
        loss1 = tf.reduce_mean(tf.square(q_vals - q1))
        loss2 = tf.reduce_mean(tf.square(q_vals - q2))
        q_loss = loss1 + loss2

    grads = tape.gradient(q_loss, q_model.trainable_variables)
    q_model.optimizer.apply_gradients(zip(grads, q_model.trainable_variables))

    # 方策の学習
    with tf.GradientTape() as tape:
        # アクションを出力
        selected_actions, means, stddevs, action_orgs = policy_model.sample_actions(states, training=True)

        # logπ(a|s) (Squashed Gaussian Policy)
        logpi = compute_logpi_sgp(means, stddevs, action_orgs)

        # Q値を出力, 小さい方を使う
        q1, q2 = q_model(states, selected_actions)
        q_min = tf.minimum(q1, q2)

        # alphaは定数扱いなので勾配が流れないようにする
        policy_loss = q_min - (tf.stop_gradient(alpha) * logpi)
        policy_loss = -tf.reduce_mean(policy_loss)  # 最大化

    grads = tape.gradient(policy_loss, policy_model.trainable_variables)
    policy_model.optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

    # パラメータαの自動調整（目的関数を最小化するような勾配を用いる）
    _, means, stddevs, action_orgs = policy_model.sample_actions(states, training=True)
    logpi = compute_logpi_sgp(means, stddevs, action_orgs)

    with tf.GradientTape() as tape:
        entropy_diff = - logpi - target_entropy
        # entropy_diffの正・負でαを減少・増加させ、調整する（勾配の大きさはentropy_diffの大きさに比例）
        log_alpha_loss = tf.reduce_mean(tf.exp(log_alpha) * entropy_diff)

    grad = tape.gradient(log_alpha_loss, log_alpha)
    q_model.optimizer.apply_gradients([(grad, log_alpha)])

    # Soft Target Update
    target_q_model.set_weights(
        (1 - soft_target_tau) * np.array(target_q_model.get_weights(), dtype=object)
        + soft_target_tau * np.array(q_model.get_weights(), dtype=object))

    # Hard Target Sync
    if all_train_count % hard_target_interval == 0:
        target_q_model.set_weights(q_model.get_weights())

    return policy_loss, q_loss


def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Pendulum環境を作成
    env = gym.make('Pendulum-v1')

    # ハイパーパラメータ
    buffer_size = 1000  # Experienceのキュー容量
    warmup_size = 500  # 学習するかどうかのExperienceの最低限の容量
    train_interval = 10  # 学習する制御周期間隔
    batch_size = 32  # バッチサイズ
    gamma = 0.9  # 割引率
    soft_target_tau = 0.02  # Soft TargetでTargetに近づく割合
    hard_target_interval = 100  # Hard Targetで同期する間隔

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

    # 記録用
    history_rewards = []
    history_metrics = []
    history_metrics_y = []

    # 学習ループ
    for episode in range(500):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        metrics_list = []

        # １エピソード
        while not done:
            # アクションを決定
            env_action, action = policy_model.sample_action(state, True)
            if isnan(env_action[0]):
                print("action is NaN. 学習失敗.")
                break
            # print("state:", state, "action:", action)

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
            if len(experiences) >= warmup_size and all_step_count % train_interval == 0:
                # モデルの更新
                metrics = update_model(
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
                metrics_list.append(metrics)
            all_step_count += 1

        # 報酬
        history_rewards.append(total_reward)

        # メトリクス
        if len(metrics_list) > 0:
            history_metrics.append(np.mean(metrics_list, axis=0))  # 平均を保存
            history_metrics_y.append(episode)

        #--- print
        interval = 20
        if episode % interval == 0:
            print("{} (min,ave,max)reward {:.1f} {:.1f} {:.1f}, alpha={:.3f}".format(
                episode,
                min(history_rewards[-interval:]),
                np.mean(history_rewards[-interval:]),
                max(history_rewards[-interval:]),
                tf.math.exp(log_alpha).numpy(),
            ))

    env.close()

    # プロット
    plt.plot(history_rewards, label="reward")
    plt.tight_layout()
    plt.xlabel('episode')
    plt.grid()
    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('episode')
    ax1.grid()
    ax1.plot(history_metrics_y, [m[0] for m in history_metrics], color="C0", marker='.', label="policy_loss")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(history_metrics_y, [m[1] for m in history_metrics], color="C1", marker='.', label="q_loss")
    ax2.legend(loc='upper right')

    fig.tight_layout()  # レイアウトの設定
    # plt.savefig('cartpole2.png') # 画像の保存
    plt.show()

    return policy_model, q_model

# テスト --5回パフォーマンス測定


def test(policy_model):
    env = gym.make('Pendulum-v1', render_mode='human')
    for episode in range(5):
        state, _ = env.reset()
        env.render()
        done = False
        total_reward = 0
        step = 0

        # １エピソード
        while not done:
            action = policy_model.sample_action(state)
            n_state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            state = n_state
            step += 1
            total_reward += reward

            done = terminated or truncated

        print("{} step, reward: {}".format(step, total_reward))
    env.close()


policy_model, _ = main()
test(policy_model)
