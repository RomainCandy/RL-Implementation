import cProfile
from ars import AgentBRS
import gym

if __name__ == "__main__":
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name)
    agent = AgentBRS(env_name=env_name, env=env, monitor_dir="./test_rofl", logger=None, norm_reward=False)
    cProfile.run("agent.learn(num_episode=3, step_size=0.02, std_noise=0.03, batch_size=8,"
                 "num_best_dir=4, threshold=200)", filename="foo2.stats")
