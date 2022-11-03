import gym

env = gym.make('Pendulum-v1', render_mode="human") # render_modeをhumanにするとwindowが現れる

observation = env.reset()
for t in range(100):
    # env.render()  # render game screen
    action = env.action_space.sample()  # this is random action. replace here to your algorithm!
    observation, reward, terminated, truncated, info = env.step(action)  # get reward and next scene
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.close()

env = gym.make('Pendulum-v1') # render_modeをhumanにしないとwindowが現れない

observation = env.reset()
for t in range(100):
    # env.render()  # render game screen
    action = env.action_space.sample()  # this is random action. replace here to your algorithm!
    observation, reward, terminated, truncated, info = env.step(action)  # get reward and next scene
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.close()