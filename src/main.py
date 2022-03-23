import copy
import pprint
import os
import time
import dill

from numpy import random
from poet.poet import POET, BruteForce, RandomSample, ReferenceApproach, POETForDMES
from poet import solver
import utils.locator

from utils import timeit_decorator as ttime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### SETUP

# BOILER_CHP EXAMPLE

# project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                             'test_cases',
#                             'boiler_chp')

# solver.PROJECT_PATH = project_path

# # agent and env params to optimise with
# agent_params = [
#     (1, 'chp_gas', 'bld1'),
#     (1, 'boiler_gas', 'bld1'),
# ]
# env_params = [
#     ('elec_supply', 'bld1', 'costs_invest_per_use'),
#     ('gas_supply', 'bld1', 'costs_invest_per_use'),
# ]

# BOILER_CHP_PV_BATTERY EXAMPLE

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_cases',
                            'boiler_chp_pv_battery')

solver.PROJECT_PATH = project_path

# agent and env params to optimise with
agent_params = [
    (1, 'chp_gas', 'bld1'),
    (1, 'boiler_gas', 'bld1'),
    (1, 'pv', 'bld1'),
    (1, 'battery', 'bld1'),
]
env_params = [
    ('elec_supply', 'bld1', 'costs_invest_per_use'),
    ('elec_supply', 'bld1', 'co2_per_use'),
    ('gas_supply', 'bld1', 'costs_invest_per_use'),
    ('gas_supply', 'bld1', 'co2_per_use'),
]

### RUN MODIFIED POET

# algo = solver.setup(project_path,
#                         env_params,
#                         agent_params,
#                         methodology=POETForDMES,
#                         load_rewards=True,
#                         verbose=False) 
# algo.load_rewards(utils.locator.get_rewards_path(project_path, "BruteForce"))
# algo = solver.solve(algo)

# algo.calc_average_R2()
# agents_ranked = list(np.array(algo.average_R2)[:,0])
# active_agents_per_iter = [[a for e,a in entry] for entry in algo.iterations_log]
# x = []
# y = []
# for i,a_list in enumerate(active_agents_per_iter[1:]):
#     x.extend([i]*len(a_list))
#     y.extend([agents_ranked.index(a) for a in a_list])

# plt.scatter(x=x,
#             y=y)
# plt.show()

# ALGO_PICKLE_PATH = os.path.join(utils.locator.get_outputs(project_path),
#                             "poet_for_visualiser.pickle")

# with open(ALGO_PICKLE_PATH, 'wb') as f:
#     dill.dump(algo, f)

### RUN COMPARISON OF METHODOLOGIES

# algo = solver.setup(project_path,
#                     env_params,
#                     agent_params,
#                     methodology=RandomSample,
#                     load_rewards=False,
#                     verbose=False)
# algo.load_rewards(utils.locator.get_rewards_path(
#             algo.project_path, "BruteForce"))
# algo.verbose = False
# algo = solver.solve(algo)
# algo.save()
# algo.calc_average_R2()
# solver.save_average_R2_scores(algo, suffix=99)


### RUN COMPARISON OF METHODOLOGIES

algo = solver.setup(project_path,
                    env_params,
                    agent_params,
                    methodology=BruteForce,
                    load_rewards=False,
                    verbose=False)
algo.load_rewards(utils.locator.get_rewards_path(
            algo.project_path, "BruteForce"))
algo.verbose  = False
algo = solver.solve(algo)
algo.save()

algos = []
for method in [BruteForce, ReferenceApproach, RandomSample, POETForDMES]:
    
    ttime.reset()
    utils.locator.create_outputs_dir(project_path)
    algo = solver.setup(project_path,
                    env_params,
                    agent_params,
                    methodology=method,
                    load_rewards=True,
                    verbose=False)
    if method is not BruteForce: 
        algo.load_rewards(utils.locator.get_rewards_path(
            algo.project_path, "BruteForce"))
        algo.verbose = False
        # algo.iterations = i
        algo = solver.solve(algo)
        if method is POETForDMES:
            print(algo.pairs_active)
            print(len(algo.rewards_cache_r2))
        algo.save()
    algo.calc_average_R2(agents=algo.df["Agent"].unique())

    # print(f"""
    #     METHOD: {str(method.__name__)}

    #     Average R2 scores:
    #     """)

    # for row in algo.average_R2:
    #     print("\t", *row[0], "\t", row[1])
    
    if method is POETForDMES: print(len(algo.rewards_cache_r2))


    algos.append(copy.deepcopy(algo))
    
    if hasattr(algo, "rewards_cache"): print(f"Num fitness evals: {len(algo.reward_cache)}")

    # ttime.print_it()

    del algo

solver.save_all_average_R2_scores(*algos, suffix=0)
solver.rank_average_R2_scores(suffix=0)

# algos[-1]._logged_rewards

#%%
### PLOT PAIRS MATRIX FOR PROPOSED

# algo_poet = solver.setup(project_path,
#                         env_params,
#                         agent_params,
#                         methodology=POETForDMES,
#                         load_rewards=True,
#                         verbose=False) 
# algo_poet.load_rewards(utils.locator.get_rewards_path(project_path, "BruteForce"))
# algo_poet = solver.solve(algo_poet)
# # df = pd.DataFrame(algo_poet._logged_rewards[1:], columns=algo_poet._logged_rewards[0])
# algo_poet.calc_average_R2()

# l = [[algo_poet.envs.index(x[0]), 
#       algo_poet.agents.index(x[1]),
#       x[6]] 
#      for x in algo_poet._logged_rewards[1:]]

# mask = np.empty(shape=(81,81))
# mask[:] = np.nan
# for e,a,i in l:
#     mask[a,e] = i

# agents_ranked = [algo_poet.agents.index(t[0]) for t in algo_poet.average_R2]

# plt.matshow(mask, cmap="Purples")
# x = []
# y = []
# for e,a in algo_poet.pairs_archive:
#     e = algo_poet.envs.index(e)
#     if algo_poet.agents.index(a) >= len(agents_ranked):
#         a = np.nan
#     else: a = agents_ranked[algo_poet.agents.index(a)]
#     x.append(e)
#     y.append(a)
# plt.scatter(x,y,c='black', s=30)
# x = []
# y = []
# for e,a in algo_poet.pairs_active:
#     e = algo_poet.envs.index(e)
#     if algo_poet.agents.index(a) >= len(agents_ranked):
#         a = np.nan
#     else: a = agents_ranked[algo_poet.agents.index(a)]
#     x.append(e)
#     y.append(a)
# plt.scatter(x,y,c='red', s=30)
# plt.legend(loc='upper left')

# plt.ylabel("Agent")
# plt.xlabel("Env")
# plt.show()

#%%
### PLOT RANKED AGENT PER ENV
# algo_poet = solver.setup(project_path,
#                         env_params,
#                         agent_params,
#                         methodology=POETForDMES,
#                         load_rewards=True,
#                         verbose=False) 
# algo_poet.load_rewards(utils.locator.get_rewards_path(project_path, "BruteForce"))
# algo_poet = solver.solve(algo_poet)

# df = pd.DataFrame(algo_poet._logged_rewards[1:], columns=algo_poet._logged_rewards[0])
# df_active = pd.DataFrame(algo_poet.pairs_active, columns=['env', 'agent'])
# df_archive = pd.DataFrame(algo_poet.pairs_archive, columns=['env', 'agent'])
# mask = np.empty(shape=(81,81))
# mask[:] = np.nan
# x_active = []
# y_active = []
# x_archive = []
# y_archive = []

# for i, env in enumerate(algo_poet.envs):
#     dff = df[df['env'] == env].sort_values(by='indicator').drop_duplicates(subset=['env', 'agent'])
#     dff_active = dff.reset_index().merge(df_active, on=['agent', 'env'])
#     ranks = [1]*len(dff.index)
#     mask[0:len(ranks),i] = ranks
#     if len(dff_active.index) > 0:
#         ls = [dff.index.get_loc(i) for i in dff_active['index']]
#         x_active.extend([i]*len(ls))
#         y_active.extend(ls)
#     dff_archive = dff.reset_index().merge(df_archive, on=['agent', 'env'])
#     if len(dff_archive.index) > 0:
#         ls = [dff.index.get_loc(i) for i in dff_archive['index']]
#         x_archive.extend([i]*len(ls))
#         y_archive.extend(ls)

# plt.matshow(mask, cmap="Purples_r")
# plt.scatter(x_active, y_active, c='red', s=30)
# plt.scatter(x_archive, y_archive, c='black', s=30)
# plt.legend(loc='upper left')

# plt.ylabel("Agent Rank")
# plt.xlabel("Env")
# plt.show()

#%%
### PLOT PAIRS MATRIX FOR REFERENCE
# algo_ref = algos[1]
# mask = np.empty(shape=(81,81))
# mask[:] = np.nan
# for i in [0,1,3,4,5,6,10,14]:
#     mask[i,:] = 1
# plt.matshow(mask, cmap='Purples_r', vmax=1)
# plt.ylabel("Agent")
# plt.xlabel("Env")
# plt.show()

### PLOT PAIRS MATRIX FOR RANDOM
# algo_random = algos[2]
# mask = np.empty(shape=(81,81))
# mask[:] = np.nan
# for env, agent in algo_random.pairs_sample:
#     e = algo_random.envs.index(tuple(env))
#     a = algo_random.agents.index(tuple(agent))
#     mask[a,e] = 1
# plt.matshow(mask, cmap='Purples_r', vmax=1)
# plt.ylabel("Agent")
# plt.xlabel("Env")
# plt.show()


### HYPERPARAMETER TUNING

# algo = solver.setup(project_path,
#                     env_params,
#                     agent_params,
#                     methodology=POETForDMES,
#                     load_rewards=True,
#                     verbose=False)
# algo.load_rewards(utils.locator.get_rewards_path(
#     algo.project_path, "BruteForce"))

# import numpy as np

# def avg_r2_active(x):
#     global algo
#     # x[0] = mutate, x[1] = diff to transfer
#     # algo.iterations = 20
#     algo.noise_generator = np.random.default_rng(seed=random.randint(0, 10000))
    
#     algo.mutation_interval = x[0]
#     algo.transfer_interval = x[1]
#     algo._optimizer_elite_size = x[2]
#     algo._optimizer_population_size = x[2]+x[3]
    
#     # for hp, val in hyperparams.items():
#     #     if hasattr(algo, hp):
#     #         setattr(algo, hp, val)
#     algo = solver.solve(algo)
#     return sum([algo.reward(e,a)[1] for e,a in algo.pairs_active]) / 1e10

# def f(x):
#     vals = np.array([avg_r2_active(x) for i in range(10)])
#     s, std = sum(vals), np.std(vals)
#     # print(vals, s, std)
#     return s

# from skopt import gp_minimize

# for i in range(10):
#     res = gp_minimize(f,      # the function to minimize
#                     [(1,10),(1,10), (1,10), (1,10)],      # the bounds on each dimension of x
#                     acq_func="EI",      # the acquisition function
#                     n_calls=10,         # the number of evaluations of f
#                     # n_random_starts=5,  # the number of random initialization points
#                     noise=0.1**2,       # the noise level (optional)
#                     # random_state=i*i
#                     )   # the random seed

#     print(res.x, res.func_vals[res["x_iters"].index(res.x)])
# # algo.save()


#################

