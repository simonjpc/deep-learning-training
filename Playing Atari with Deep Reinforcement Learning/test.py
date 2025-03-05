import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    episode_over = terminated or truncated

env.close()

"""

Notes on Pong useful actions:

0: NOOP
2: UP
5: DOWN

Source: https://ale.farama.org/env-spec/
"""