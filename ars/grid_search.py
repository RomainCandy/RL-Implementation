from collections import namedtuple
from multiprocessing import Pool
from ars import run_experiment
import pybullet_envs


Search = namedtuple("Search", ["env_name", "threshold", "step_size", "std_noise", "batch_size",
                               "num_best_dir", "seed", "norm_reward"])

BidepSearch = Search(env_name="BipedalWalker-v2", threshold=300, step_size=[0.02],
                     std_noise=[0.03], batch_size=[8, 8, 16, 16, 32, 32], num_best_dir=[4, 8, 8, 16, 16, 32],
                     seed=[16, 32, 64], norm_reward='clip')


BidepHardSearch = Search(env_name="BipedalWalkerHardcore-v2", threshold=300, step_size=[0.015, 0.02],
                         std_noise=[0.025, 0.03], batch_size=[128, 128, 128],
                         num_best_dir=[32, 64, 128], seed=[128], norm_reward='clip')


LunarSearch = Search(env_name="LunarLanderContinuous-v2", threshold=200, step_size=[0.02],
                     std_noise=[0.03], batch_size=[8, 8, 16, 16, 32, 32], num_best_dir=[4, 8, 8, 16, 16, 32],
                     seed=[4, 8], norm_reward='id')


PendulumSearch = Search(env_name="Pendulum-v0", threshold=-160, step_size=[0.02],
                        std_noise=[0.03], batch_size=[8, 8, 16, 16, 32, 32], num_best_dir=[4, 8, 8, 16, 16, 32],
                        seed=[4, 8], norm_reward='id')

HopperSearch = Search(env_name="HopperBulletEnv-v0", threshold=2500.0, step_size=[0.02],
                      std_noise=[0.03], batch_size=[8], num_best_dir=[4],
                      seed=[5, 9, 17], norm_reward='shift')


CheetahSearch = Search(env_name="HalfCheetahBulletEnv-v0", threshold=3000.0, step_size=[0.02],
                       std_noise=[0.03], batch_size=[32], num_best_dir=[16],
                       seed=[11, 111, 1111], norm_reward='shift')


AntSearch = Search(env_name="AntBulletEnv-v0", threshold=2500.0, step_size=[0.015],
                   std_noise=[0.025], batch_size=[64], num_best_dir=[32],
                   seed=[4444, 44, 444], norm_reward='shift')


WalkerSearch = Search(env_name="Walker2DBulletEnv-v0", threshold=2500.0, step_size=[0.025],
                      std_noise=[0.01], batch_size=[60], num_best_dir=[60],
                      seed=[6, 66, 666], norm_reward='id')

ThrowerSearch = Search(env_name="ThrowerBulletEnv-v0", threshold=18, step_size=[0.03],
                       std_noise=[0.025], batch_size=[32], num_best_dir=[16],
                       seed=[2, 22, 222], norm_reward='id')

InvertedDoublePendulum = Search(env_name="InvertedDoublePendulumBulletEnv-v0", threshold=9100, step_size=[0.03],
                                std_noise=[0.025], batch_size=[32], num_best_dir=[16],
                                seed=[3, 33, 333], norm_reward='id')


def get_all(envSearch: Search):
    seed = envSearch.seed
    env_name = envSearch.env_name
    thresh = envSearch.threshold
    step_size = envSearch.step_size
    std_noise = envSearch.std_noise
    batch_size = envSearch.batch_size
    num_best_dir = envSearch.num_best_dir
    norm_reward = envSearch.norm_reward

    res = list()
    i = 0
    for st in step_size:
        for noise in std_noise:
            for s in seed:
                for bs, nbd in zip(batch_size, num_best_dir):
                    i += 1
                    res.append(("videos", 50000, s, env_name, st, noise, bs, nbd, thresh, norm_reward))
    return res


# exp = get_all(BidepSearch)

def first_exp(tup):
    return run_experiment(*tup)


def my_exp(tup):
    return run_experiment(*tup, load=True)


# with Pool(3) as p:
#     p.map(my_exp, exp)

# exp = get_all(BidepHardSearch)
# exp = get_all(AntSearch)
# exp = get_all(CheetahSearch)
with Pool(3) as p:
    # exp = get_all(HopperSearch)
    # p.map(my_exp, exp)
    # exp = get_all(AntSearch)
    exp = get_all(WalkerSearch)
    # exp = get_all(InvertedDoublePendulum)
    p.map(first_exp, exp)
