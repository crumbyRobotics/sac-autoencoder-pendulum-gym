import gym

env = gym.make('Pendulum-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            break
