
import os
from typing import Type

import dill
import toml
from utils import locator, validation

from . import config
import pandas as pd
import numpy as np

import ehub
from poet import poet

PROJECT_PATH = None
COLUMNS_AVERAGE_R2 = ["Agent", 
                      "Average_R2_BruteForce", "Std_R2_Bruteforce",
                      "Average_R2_StateOfTheArt", "Std_R2_StateOfTheArt",
                      "Average_R2_Random", "Std_R2_Random",
                      "Average_R2_POET", "Std_R2_POET"]


def setup(project_path,
          env_keys,
          agent_keys,
          methodology: Type[poet.POETForDMES] = poet.POETForDMES,
          verbose=False,
          **kwargs) -> poet.POETForDMES:  # just for a network for now

    global PROJECT_PATH
    validation.mode = validation.Mode.BOUNDED
    PROJECT_PATH = project_path
    ehub.VERBOSE = verbose

    # Initialise project
    project = ehub.Project(project_path)
    network = create_fixed_network(project)
    hyperparameters = get_poet_hyperparameters(project_path)
    c = config.Config(network=network,
                      agent_keys=agent_keys,
                      env_keys=env_keys)  # could use agent and env as input

    algo = methodology(config=c, **kwargs, **hyperparameters)
    return algo


def solve(algo: poet.POETForDMES,
          load=False) -> poet.POETForDMES:
    if load:
        # TODO
        with open(locator.get_rewards_path(algo.project_path, "ehubFixedNetworkPOET"), 'rb') as file:
            algo.rewards = dill.loads(file.read())
    else:
        algo.run()
    return algo


def plot(algo,
         env,
         agent):
    # PLOT
    algo.plot(env, agent)
    return algo


def save_all_average_R2_scores(algo_bruteforce: poet.BruteForce,
                           algo_stateoftheart: poet.ReferenceApproach,
                           algo_random: poet.RandomSample,
                           algo_poet: poet.POETForDMES,
                           suffix=None):

    dfs = [
        pd.DataFrame(algo_bruteforce.average_R2, columns=[
                     "Agent", "Average_R2_BruteForce", "Std_R2_Bruteforce"]),
        pd.DataFrame(algo_stateoftheart.average_R2, columns=[
                     "Agent", "Average_R2_StateOfTheArt", "Std_R2_StateOfTheAart"]),
        pd.DataFrame(algo_random.average_R2,
                columns=["Agent", "Average_R2_Random", "Std_R2_Random"]),
        pd.DataFrame(algo_poet.average_R2,
                     columns=["Agent", "Average_R2_POET", "Std_R2_POET"])
    ]
    result = pd.merge(dfs[0], dfs[1], how='outer', on="Agent")
    result = pd.merge(result, dfs[2], how='outer', on="Agent")
    result = pd.merge(result, dfs[3], how='outer', on="Agent")
    result.to_csv(locator.get_optimality_path(PROJECT_PATH, num=suffix))
    
def save_average_R2_scores(algo: poet.POETForDMES, suffix=None):
    pd.DataFrame(algo.average_R2,
                     columns=["Agent", "Average_R2_POET", "Std_R2_POET"])\
        .to_csv(locator.get_optimality_path(PROJECT_PATH, num=suffix))


def rank_average_R2_scores(save=True, suffix=0):
    df = pd.read_csv(locator.get_optimality_path(PROJECT_PATH, num=suffix))

    ranks = pd.DataFrame.from_dict(dict(zip(tuple(COLUMNS_AVERAGE_R2),
                                            [list(df["Agent"])] +
                                            [list(np.argsort(df[col]))
                                             for col in df.columns[2:]]
                                            )))
    if save:
        ranks.to_csv(locator.get_optimality_ranked_path(PROJECT_PATH, num=suffix))
    return ranks


def get_poet_hyperparameters(project_path):
    with open(locator.get_poet_hyperparameters(project_path), 'r') as fp:
        return toml.loads(fp.read())


def create_fixed_network(project: ehub.Project):

    def get_fixed_network(project_path):
        with open(locator.get_fixed_network(project_path), 'r') as fp:
            return toml.loads(fp.read())

    network = ehub.Network(project)
    network_config = get_fixed_network(project.project_path)
    network.set_network_config(network_config)
    return network


def update_reward(finished_poet: poet.POET, agent, env):
    return finished_poet.reward(env, agent)
