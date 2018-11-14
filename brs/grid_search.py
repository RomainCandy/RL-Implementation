from collections import namedtuple
from multiprocessing import Pool
from brs import run_experiment
# import os


Search = namedtuple("Search", ["env_name", "threshold", "step_size", "std_noise", "batch_size",
                               "num_best_dir", "seed"])

BidepSearch = Search(env_name="BipedalWalker-v2", threshold=300, step_size=[0.01, 0.02],
                     std_noise=[0.01, 0.02], batch_size=[4, 8, 16], num_best_dir=[2, 4, 8], seed=[1946])


def get_all(envSearch: Search):
    seed = envSearch.seed
    env_name = envSearch.env_name
    thresh = envSearch.threshold
    step_size = envSearch.step_size
    std_noise = envSearch.std_noise
    batch_size = envSearch.batch_size
    num_best_dir = envSearch.num_best_dir

    res = list()
    i = 0
    for st in step_size:
        for noise in std_noise:
            for s in seed:
                for bs, nbd in zip(batch_size, num_best_dir):
                    i += 1
                    res.append(("video{}_seed{}".format(i, s), 3000, s, env_name, st, noise, bs, nbd, thresh))
    return res


exp = get_all(BidepSearch)


def my_exp(tup):
    return run_experiment(*tup)


with Pool(3) as p:
    p.map(my_exp, exp)

# print(os.path.abspath('.'))
