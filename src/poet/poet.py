import csv
import itertools
import os
import pickle
from enum import Enum
from typing import OrderedDict, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import toml
import dill
from numpy.core.fromnumeric import mean
from pyomo.core.base.PyomoModel import ConcreteModel

from . import r2_indicator as r2
from . import config
import lp.model
import lp.solver

import utils.locator
import utils.timeit_decorator as ttime
import visualiser.visualiser as vis

class OptimiserGradientEnum(Enum):
    ES_Elite = 0


class Optimizer():

    def __init__(self, project_path) -> None:
        self.project_path = project_path
        self.rewards = []
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def load(self):
        # try:
        #     with open(utils.locator.get_outputs(self.project_path), 'rb') as f:
        #         algo = dill.loads(f.read())
        #     print("POET pickled")
        #     algo.setup_reward_model()  # need to reset pyomo model to match ids
        #     algo.save()
        #     return algo
        # except:
        #     print("No pickle found, running debug...")
        #     algo = methodology(config=c,**kwargs, **hyperparameters)
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def tester(self, env, agent) -> Tuple[list, float]:
        r, i = self._reward(env, agent)
        return r, i

    def _reward(self):
        raise NotImplementedError


class POET(Optimizer):

    def __init__(self,
                 config: config.Config,
                 load_data: bool = True,
                 verbose: bool = False,
                 # HYPERPARAMS
                 # what to use for calc reward function
                 optimiser_gradient: OptimiserGradientEnum = OptimiserGradientEnum.ES_Elite,
                 learning_rate: float = 0.01,
                 noise_stdev: float = 0.001,
                 iterations: int = 20,
                 mutation_interval: int = 10,
                 transfer_interval: int = 10,
                 max_children: int = 100,
                 max_admitted: int = 100,
                 capacity: int = 100,
                 min_reward_reproduce_agent: float = 100.0,
                 min_reward_keep_env: float = 50.0,
                 max_reward_keep_env: float = 200.0,
                 minimise_reward: bool = True,
                 optimizer_population_size: int = 40,
                 optimizer_elite_size: int = 10,
                 novelty_k: int = 5,
                 random_seed: int = 42,
                 ) -> None:

        # Setup config
        self.config = config
        self.verbose = verbose

        # Setup optimisers
        if optimiser_gradient in OptimiserGradientEnum:
            self.optimiser_gradient = optimiser_gradient  # TODO implement
        else:
            raise NotImplementedError()

        # Setup pairs
        self.pairs_active = []
        self.pairs_archive = []  # archive of inactive env-agent pairs, could use id as key
        self.reward_cache = OrderedDict()

        # Noise parameters
        self.noise_stdev = abs(noise_stdev)
        self.noise_sample_count = 1000
        self.noise_generator = np.random.default_rng(seed=random_seed)
        self._setup_noise_probabilities()

        # Optimiser parameters
        self.learning_rate = learning_rate
        self._optimizer_population_size = optimizer_population_size
        self._optimizer_elite_size = optimizer_elite_size
        assert self._optimizer_elite_size <= self._optimizer_population_size, \
            f"Elite size ({self._optimizer_elite_size}) must be smaller \
                than population size ({self._optimizer_population_size})."
        # check that you can create a population of different agents given the std dev and size
        assert (self.noise_stdev*2+1) * self.config.agent_size >= self._optimizer_population_size, \
            "Possible agent mutation combinations is less than population size."

        # Assert that stddev is not wider than max-min % step
        for lb, ub, step in zip(self.config._agent_lower, self.config._agent_upper, self.config._agent_step):
            assert (self.noise_stdev*2) <= int((ub-lb) / step), \
                f"Noise deviation {self.noise_stdev} is wider than allowed bounds and steps of the agent parameter:" + \
                f"\n\tLower = {lb} \n\tUpper = {ub} \n\tStep = {step} \n\tNumber of Steps = {int((ub-lb) / step)}"
        self._prepare_noise_combinations()

        # Iteration and interval parameters
        self.iterations = iterations
        self.mutation_interval = mutation_interval
        self.transfer_interval = transfer_interval

        self.MAX_CHILDREN = max_children  # maximum number of children per reproduction
        # maximum number of children admitted per reproduction
        self.MAX_ADMITTED = max_admitted
        self.CAPACITY = capacity  # maximum number of active environments

        # Minimum condiion for repoducing an agent
        assert len(min_reward_reproduce_agent) == self.config.num_objectives, f"MIN_REWARD_REPRODUCE_AGENT must be an array of same length as number of objectives."
        assert len(min_reward_keep_env) == self.config.num_objectives, f"MIN_REWARD_KEEP_ENV must be an array of same length as number of objectives."
        assert len(max_reward_keep_env) == self.config.num_objectives, f"MAX_REWARD_KEEP_ENV must be an array of same length as number of objectives."

        # minimum reward to allow agent to reproduce
        self.MIN_REWARD_REPRODUCE_AGENT = np.array(min_reward_reproduce_agent)
        # lower bound for minimum criterium to keep a generaed env
        self.MIN_REWARD_KEEP_ENV = np.array(min_reward_keep_env)
        # upper bound for minimum criterium to keep a generaed env
        self.MAX_REWARD_KEEP_ENV = np.array(max_reward_keep_env)

        self.minimise_reward = minimise_reward

        # Novelty parameters
        self.NOVELTY_K = novelty_k  # number of neighbors for kNN calc

        # For logging
        self._current_iteration = 0
        self._logged_rewards = [
            ['env', 'agent', 'reward', 'indicator', 'agent_parent', 'env_parent', 'iteration']]

    @ttime.timeit
    def _prepare_noise_combinations(self):
        agent_noise = [
            range(-self.noise_stdev, self.noise_stdev + 1)] * self.config.agent_size
        self._optimizer_agent_mutations = list(itertools.product(*agent_noise))
        self._optimizer_agent_mutations_size = len(
            self._optimizer_agent_mutations)  
        env_noise = [
            range(-self.noise_stdev, self.noise_stdev + 1)] * self.config.env_size
        self._optimizer_env_mutations = list(itertools.product(*env_noise))
        self._optimizer_env_mutations_size = len(self._optimizer_env_mutations)

    @ttime.timeit
    def _setup_noise_probabilities(self):
        x = np.arange(-self.noise_stdev, self.noise_stdev + 1)
        prob = np.round(np.random.normal(scale=3, size=10))
        # normalize the probabilities so their sum is 1
        self._noise_probability = prob / prob.sum()

    @ttime.timeit
    def eligible_to_reproduce(self, env, agent) -> bool:
        r, _ = self.reward(env, agent)
        for r_i in r:
            if (r_i > self.MIN_REWARD_REPRODUCE_AGENT).all():
                return False
        return True

    @ttime.timeit
    def env_reproduce(self, pairs: list) -> list:
        pairs_new = []
        for env, agent in pairs:
            noise = self._sample_noise_env(env)
            env_new = tuple(round(e + n * s, ndigits=2)
                            for e, n, s in zip(env, noise, self.config._env_step))
            pairs_new.append((env_new, agent))
        return pairs_new

    @ttime.timeit
    def euclidean_distance(self, x, y):  # COPIED from POET
        x = np.array(x)
        y = np.array(y)
        n, m = len(x), len(y)
        if n > m:
            a = np.linalg.norm(y - x[:m])
            b = np.linalg.norm(y[-1] - x[m:])
        else:
            a = np.linalg.norm(x - y[:n])
            b = np.linalg.norm(x[-1] - y[n:])
        return np.sqrt(a**2 + b**2)

    @ttime.timeit
    def rank_by_novelty(self, pairs_candidate) -> list:
        """
        """

        def novelty(env) -> float:  # COPIED from POET -> novely.compute_novelty_vs_archive
            distances = []
            # todo: implement PATA-EC
            for env_archived, _ in self.pairs_archive:
                distances.append(self.euclidean_distance(env, env_archived))

            for env_candidate, _ in pairs_candidate:
                distances.append(self.euclidean_distance(env, env_candidate))

            # Pick k nearest neighbors
            k = self.NOVELTY_K
            distances = np.array(distances)
            top_k_indicies = (distances).argsort()[:k]
            top_k = distances[top_k_indicies]
            return mean(top_k)  # top_k.mean()

        envs_archived = [env for env, _ in self.pairs_archive]

        return sorted(pairs_candidate, key=lambda pair: novelty(pair[0]))

    @ttime.timeit
    def optimize_step(self, env, agent_start):
        """Performs local optimisation of agent in env
        returns the modified agent

        Given start point, population size, 
        Generate population by adding noise to start point
        Calc reward for each individual
        Calc distance from start point to each individual
        Multiply distances by rewards
        Sum all of these into one vector * learning rate?
        """

        def calc_gradient(vectors):
            noise_gradient = np.round(np.sum(vectors, axis=0))
            agent_with_gradient = []
            for i, t in enumerate(zip(agent_start, 
                                  noise_gradient, 
                                  self.config._agent_step,
                                  )):
                a, n, step = t
                g = a + n*step
                if g < self.config._agent_lower[i]:
                    agent_with_gradient.append(self.config._agent_lower[i])
                elif g > self.config._agent_upper[i]:
                    agent_with_gradient.append(self.config._agent_upper[i])
                else:
                    agent_with_gradient.append(g)
            # agent_with_gradient = tuple(a + step*g for a, step,
            #              g in zip(agent_start, 
            #                       self.config._agent_step, 
            #                       gradient))
            return tuple(agent_with_gradient)

        _, indicator_start = self.reward(
            env, agent_start, agent_parent=agent_start)
        if self.verbose:
            print(
                f"Optmising locally...\tEnv: {env}\tAgent: {agent_start}\tIndicator: {indicator_start}")

        # Calculate the rewards of noisy agents
        noises = self._sample_noise_agent(agent_start)
        self._optimizer_current_pair = (env, agent_start)
        indicators = self._get_indicators(noises)

        # If not successful, try once more. Otherwise, return agent_start
        if indicators is None or len(indicators) == 0:
            noises = self._sample_noise_agent(agent_start)
            indicators = self._get_indicators(noises)
            if not indicators:
                return agent_start

        del self._optimizer_current_pair

        # find the min indicator for normalising
        indicator_best = indicator_start - min(indicators)
        # invert norm because we want wieght base don minimum indicator
        indicators_norm = [1-i for i in (indicators / np.linalg.norm(indicators))]
        # sort indicators and pick elites
        reward_indicators_ranks = np.array(indicators).argsort()
        if len(reward_indicators_ranks) > self._optimizer_elite_size:
            reward_indicators_ranks = reward_indicators_ranks[:self._optimizer_elite_size]

        weighted_vectors = [np.array(noises[rr]) * indicators_norm[rr]*self.learning_rate #(indicator_start-indicators[rr])/indicator_best
                            for rr in reward_indicators_ranks]

        agent_with_gradient = calc_gradient(weighted_vectors)
        if self.verbose:
            print(f"\tOptimised Agent: {agent_with_gradient}")
        return agent_with_gradient

        # for plotting the vectors...

        # import matplotlib.pyplot as plt

        # V_norm = np.array(weighted_vectors)
        # origin = np.array([0]*population_size) # origin point

        # plt.quiver(origin, origin, V_norm[:,0], V_norm[:,1], color=['b'], scale=10)
        # plt.quiver(np.array([0]), np.array([0]), [gradient[0]], [gradient[1]], color=['r'], scale=20)
        # plt.show()

        # TODO replace with cached rewards for the env
        # X_init= [agent]
        # Y_init = [-self.reward(env,agent)]

        # bounds = [[-10,10]]

        # r = gp_minimize(lambda x: -self.reward(env, tuple(x)),
        #                 bounds,
        #                 acq_func='EI',      # expected improvement
        #                 xi=0.01,            # exploitation-exploration trade-off
        #                 n_calls=num_iterations,         # number of iterations
        #                 n_random_starts=0,  # initial samples are provided
        #                 x0=X_init,         # initial samples
        #                 y0=Y_init)

        # return r.x_iters, r.func_vals

    def _get_indicator(self, noise):
        env, agent_start = self._optimizer_current_pair
        # create noisy agent
        agent = tuple(a + round(step*n, ndigits=3) for a, step,
                      n in zip(agent_start, self.config._agent_step, noise))

        _, indicator_start = self.reward(env, agent_start, log=False)
        _, indicator = self.reward(env, agent, agent_parent=agent_start)

        return indicator
        # indicator = indicator_start-1

        if indicator < indicator_start:
            return indicator
        else:
            return None

    def _get_indicators(self, noises):
        indicators = [self._get_indicator(noise) for noise in noises]
        return [i for i in indicators if i is not None]

    @ttime.timeit
    def evaluate_candidates(self, agents, target_env):
        """Evaluates the candidate agents within a target environment.
        Returns the top performing agent
        """
        top_agent = agents[0]
        _, top_indicator = self.reward(target_env, top_agent)
        for agent in agents:
            # optimise agent locally by one step
            agent_optimised = self.optimize_step(target_env, agent)
            _, agent_indicator = self.reward(target_env, agent)
            _, agent_optimised_indicator = self.reward(
                target_env, agent_optimised)

            # replace top_agent with agent with smallest r2
            if agent_indicator < top_indicator:
                top_agent = agent
            if agent_optimised_indicator < top_indicator:  # assume optimised agent always better than agent
                top_agent = agent_optimised

        return top_agent

    @ttime.timeit
    def min_criterion_satisfied(self, child_list):
        """Filters through the newly generated child environmnet, agent pairs to see
        which ones are either too easy or too hard based on the minimal criterion."""
        if self.verbose:
            print("Checking min criterion...")
        filtered_list = []
        for env, agent in child_list:
            reward, _ = self.reward(env, agent)
            if self.verbose:
                print(f"\tEnv: {env}\tAgent: {agent}")
            add_to_filter = False
            for r in reward:
                # a bit confusing, but in our case a big reward is low cost,
                # so to pick environments that are not too easy or hard
                # we find rewards above a minimum reward (or below a certain cost) --> not too hard
                # and below a maximum reward (or above a certain cost) --> not too easy
                if (r < self.MIN_REWARD_KEEP_ENV).all() \
                        and (r > self.MAX_REWARD_KEEP_ENV).all():
                    add_to_filter = True
                    break
            if add_to_filter:
                filtered_list.append((env, agent))
        return filtered_list

    @ttime.timeit
    def remove_oldest(self, pairs) -> list:
        """Removes oldest env agent pairs."""
        M = len(pairs)
        if M > self.CAPACITY:
            num_removals = M - self.CAPACITY
            for p in pairs[:num_removals]:
                self.pairs_archive.append(p)
            return pairs[num_removals:]
        else:
            return pairs

    @ttime.timeit
    def mutate_envs(self) -> list:
        """Mutates the environments by checking mutation eligbility first.
        """
        pairs = self.pairs_active
        agents = [agent for _, agent in pairs]

        parent_list = [p for p in pairs if self.eligible_to_reproduce(*p)]

        child_list = self.env_reproduce(parent_list)
        child_list = self.min_criterion_satisfied(child_list)
        child_list = self.rank_by_novelty(child_list)
        admitted = 0

        for env_child, agent_child in child_list:
            agent_child = self.evaluate_candidates(agents,
                                                   env_child)
            if self.min_criterion_satisfied([(env_child, agent_child)]):
                pairs.append((env_child, agent_child))
                admitted += 1
                if admitted >= self.MAX_ADMITTED:
                    break

        pairs = self.remove_oldest(pairs)

        return pairs

    @ttime.timeit
    def reward(self, env, agent,
               agent_parent=None, env_parent=None,
               iteration=None,
               log=True):
        """
        Calculates the rewards and the scalar indicator.
        When parent_agent is not None, env is fixed for group of agents being evalutaed.
        Likewise when parent_env is not None, agent is fixed for group fo envs being evaluated.
        """
        # Calc the reward and indicator
        if (env, agent) not in self.reward_cache:
            reward, indicator = self._reward(env, agent)
            self.reward_cache[(env, agent)] = (reward, indicator)
        else:
            reward, indicator = self.reward_cache[(env, agent)]
            if self.verbose:
                print(
                    f"\tCached LP...\tAgent: {agent}\tIndicator: {indicator}")

        
        if log:
            self.log_reward(env, agent,
                            reward, indicator,
                            agent_parent, env_parent,
                            iteration)
        
        return reward, indicator

    def _reward(self, env, agent):
        raise NotImplementedError()

    def log_reward(self, env, agent,
                   reward, indicator,
                   agent_parent, env_parent,
                   iteration,
                   dump_to_file=False):
        # if dump_to_file:
        #     with open('pathtocsv.csv', 'w') as f:
        #         writer = csv.writer(f)
        #         writer.writerows(self._logged_rewards)
        #     self._logged_rewards = []
        # else:
        # if len(self._logged_rewards) == 0:
        #     self._logged_rewards = ['env', 'agent', 'reward', 'indicator', 'iteration', 'env_parent', 'agent_parent']

        if not iteration:
            iteration = self._current_iteration

        self._logged_rewards.append([env, agent,
                                     reward, indicator,
                                     agent_parent, env_parent,
                                     iteration])

    @ttime.timeit
    def run(self, print_results=True, save_results=True):
        # pairs_active = self.pairs_active
        self.pairs_active.append(
            (self.config.init_env, self.config.init_agent))
        
        self.iterations_log = []
        self.iterations_log.append(self.pairs_active)

        for t in range(self.iterations):

            if self.verbose: print(f"\n\nITERATION {t}")
            self._current_iteration = t

            # if time to mutate, mutate the environments of pairs
            if t > 0 and t % self.mutation_interval == 0:
                self.pairs_active = self.mutate_envs()

            # create next agents by optimising one step locally
            pairs_next = [(e, self.optimize_step(e, a))
                          for e, a in self.pairs_active]
            agents_next = [a for _, a in pairs_next]

            for m, pair in enumerate(pairs_next):
                env, agent_next = pair
                # check if time to attempt transfer
                if len(self.pairs_active) > 1 and t % self.transfer_interval == 0:

                    # find top of next_agents in env, except the next_agent currently assigned
                    agent_top = self.evaluate_candidates([tuple(l) for l in
                                                          agents_next[:m] + agents_next[m+1:]], env)

                    # transfer into env if it performs better than the one optimised locally
                    _, indicator_agent_top = self.reward(
                        env, agent_top, agent_parent=agent_next, iteration=t)
                    _, indicator_agent_next = self.reward(
                        env, agent_next, agent_parent=agent_next, iteration=t)

                    if indicator_agent_top < indicator_agent_next:
                        if self.verbose:
                            print(f'Transfer...replace\t{agent_next} with\t{agent_top} in \t{env}')
                        agent_next = agent_top

                self.pairs_active[m] = (env, agent_next)
            if self.verbose: print(
                f"\nPairs Active\t{len(self.pairs_active)}\n\t{self.pairs_active}\nPairs Archived\t{len(self.pairs_archive)}")
            self.iterations_log.append(self.pairs_active)
        if print_results:
            self.print()
        if save_results:
            self.save()

    @ttime.timeit
    def print(self):
        # print / save results.
        try:
            print("Active pairs: ", self.pairs_active)
            print("Archived pairs: ", self.pairs_archive)
            print("Rewards calculated: ", len(self.reward_cache))
            print("Rewards: \n")
            [print(f"Env {t[0]}: \tAgent {t[1]} ") for t in self.reward_cache]
        except ValueError:
            print("Must run() first.")

    @ttime.timeit
    def save(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def _get_mutations(self, param, config):
        """Samples from normal distribution."""
        # mean = 0, stdev based on hyperparam, shape based on lenght of agent encoding and num of samples
        noises_ub = []
        noises_lb = []
        for p, p_config in zip(param, config):
            _, lb, ub, step = p_config
            noise_lb = int((lb - p)/step)
            noise_ub = int((ub - p)/step)
            noises_lb.append(noise_lb if noise_lb > -
                             self.noise_stdev else -self.noise_stdev)
            noises_ub.append(noise_ub if noise_ub <
                             self.noise_stdev else self.noise_stdev)

        noises = []
        for lb, ub, in zip(noises_lb, noises_ub):
            noises.append(range(lb, ub + 1))
        mutations = list(itertools.product(*noises))

        return mutations

    @ttime.timeit
    def _sample_noise_agent(self, agent):
        """Shuffles the mutation combos and picks a subset based on the population size"""
        agent_mutations = self._get_mutations(agent, self.config.agent)
        assert len(
            agent_mutations) > 0, f"Agent could not be mutated. Agent: {agent}."
        self.noise_generator.shuffle(agent_mutations)
        result = list(agent_mutations[:self._optimizer_population_size])
        del agent_mutations
        return result

    @ttime.timeit
    def _sample_noise_env(self, env):
        """Shuffles the mutation combos and picks a subset based on the population size"""
        env_mutations = self._get_mutations(env, self.config.env)
        self.noise_generator.shuffle(env_mutations)
        return env_mutations[0]


class POETForDMES(POET):

    DF_COLUMNS = ["Env", "Agent", "Results", "R2"]

    def __init__(self,
                 config: config.Config,
                 optimiser_gradient: OptimiserGradientEnum = OptimiserGradientEnum.ES_Elite,
                 load_rewards: bool = False,
                 save_models: bool = True,
                 **hyperparams):

        super().__init__(config, optimiser_gradient=optimiser_gradient, **hyperparams)
        self.project_path = self.config.network._project.project_path
        self.save_models = save_models
        self.reward_save_path = utils.locator.get_rewards_path(
            self.project_path, type(self).__name__)
        if load_rewards:
            self.load()
            self.agents_simulated = self.df["Agent"].unique()
        else:
            self.rewards = []
            
        self.rewards_cache_r2 = dict()

        assert self.config.is_network, "Config must have a network."
        self._r2_utopia = [0]*len(self.config.objectives)

        # Setup Pyomo Model
        self.setup_reward_model()
        
        # Store all possible agents and envs
        agent_params = [list(np.arange(l, u+s, s))
                        for l, u, s
                        in zip(self.config._agent_lower,
                               self.config._agent_upper,
                               self.config._agent_step)]
        env_params = [list(np.arange(l, u+s, s))
                      for l, u, s
                      in zip(self.config._env_lower,
                             self.config._env_upper,
                             self.config._env_step)]

        self.agents = list(itertools.product(*agent_params))
        self.envs = list(itertools.product(*env_params))

    @ttime.timeit
    def setup_reward_model(self):
        self._reward_model = lp.model.set_up(self.config.network,
                                             self.config.objectives)

    def reward(self, env, agent, **kwargs):
        r, i = super().reward(env, agent, **kwargs)
        self.rewards_cache_r2[(env,agent)] = (r,i)
        return r, i

    @ttime.timeit
    def run(self, print_results=False):
        super().run(print_results=print_results)

        data = [[*k, *v] for k, v in self.rewards_cache_r2.items()]
        self.df = pd.DataFrame(data, columns=self.DF_COLUMNS)
        self.agents_simulated = self.df["Agent"].unique()

    @ttime.timeit
    def _reward(self, env, agent):
        if self.verbose:
            print(f"\tRunning LP...\tAgent: {agent}", end="")

        self.setup_reward_model()
        model = self._reward_model

        # sets the agent parameters in the network and in the pyomo LP model
        for n, a in zip(self.config.agent_map, agent):
            n.node_capacity_max = a
            getattr(model, self.config._cap_keys['max'])[id(n)] = a

        # sets the env parameters in the network and in the pyomo LP model
        for attr, e in zip(self.config.env_map, env):
            for loc in attr.obj.locations.value:
                attr.value[loc] = e
                getattr(model, attr.name)[(attr.obj.name, loc)] = e

        results, instances = lp.solver.solve(instance=model,
                                                      objectives=self.config.objectives,
                                                      n_points=3,
                                                      verbose=self.verbose)

        if self.save_models:
            self._save_reward_model(env, agent, instances)

        results = np.array(results)
        indicator = r2.get_r2_indicator(results,
                                        weights_start=results[0],
                                        weights_end=results[-1],
                                        utopia=self._r2_utopia)

        if self.verbose:
            print(f"\tIndicator: {indicator}")

        return results, indicator

    @ttime.timeit
    def _save_reward_model(self, env, agent, instances):
        utils.locator.create_model_dir(self.project_path, env, agent)

        for i, instance in enumerate(instances):
            instance_pickle_path = utils.locator.get_model_instance_path(
                self.project_path, env, agent, i)
            path_pickle = utils.locator.get_model_results_path(
                self.project_path, env, agent, i, ext="pickle")
            path_toml = utils.locator.get_model_results_path(
                self.project_path, env, agent, i, ext="toml")

            id_to_n_dict = {id(n): n.no_cap for n in self.config.network.nodes}

            flows_nodes = np.array([(i[0], v.value)
                                    for i, v in instance.flow_node.items()])
            flows_nodes = flows_nodes.reshape(
                len(instance.nodes), len(instance.time)+1, 2)
            flows_nodes = {str(id_to_n_dict[node]): flow
                           for node, flow in zip(
                instance.nodes.value_list,
                flows_nodes[:, :, -1])}

            with open(instance_pickle_path, 'wb') as file:
                dill.dump(instance, file)

            with open(path_pickle, 'wb') as file:
                pickle.dump({k: v for k, v in flows_nodes.items()}, file)

            with open(path_toml, 'w') as file:
                toml.dump({k: v for k, v in flows_nodes.items()}, file)

    @ttime.timeit
    def calc_average_R2(self, agents=[], consider_probabilities=False):
        df = self.df
        avgs = []

        if not list(agents):
            agents = self.agents

        for agent in agents:
            avg = np.average(df[df["Agent"] == agent]["R2"])
            std = np.std(df[df["Agent"] == agent]["R2"])
            if avg > 0.0:
                avgs.append([agent, avg, std])

        self.average_R2 = sorted(avgs, key=lambda x: x[1])

    def load(self):
        try:
            with open(self.reward_save_path, 'rb') as file:
                self.rewards = pickle.loads(file.read())
             
        except FileNotFoundError:
            print("Could not find file")
            self.rewards = []
        finally:
            self.df = pd.DataFrame(self.rewards, columns=self.DF_COLUMNS)
    
    def load_rewards(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                rewards = pickle.loads(file.read())
            self.reward_cache = OrderedDict(
                { (e,a) : (r, i)
                 for e, a, r, i
                 in rewards }
            )

    def load_model(self, env, agent) -> ConcreteModel:
        del self._reward_model
        with open(utils.locator.get_model_instance_path(self.project_path, env, agent, num=0), 'rb') as file:
            self._reward_model = dill.loads(file.read())

    def save(self) -> None:
        with open(self.reward_save_path, 'wb') as file:
            pickle.dump(self.rewards, file)

    def save_rewards(self, to_csv=True, to_pickle=True):
        if to_csv:
            with open(self.reward_save_path + '.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(
                    csvfile, delimiter=';', 
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
                header = self.config.rewards_save_header
                csvwriter.writerow(header)
                for k, v in self.reward_cache.items():
                    env, agent = k
                    reward, indicator = v
                    csvwriter.writerow([*env, *agent, reward, indicator])
            print("Saved to csv!")
        if to_pickle:
            import pickle
            data = []
            # Rearrange cache for dataframe
            for k, v in self.reward_cache.items():
                env, agent = k
                reward, indicator = v
                data.append([env, agent, reward, indicator])
            df = pd.DataFrame(data,  columns=["Env", "Agent", "Results", "R2"])
            with open(self.reward_save_path + '.pickle', 'wb') as file:
                pickle.dump(df, file)
            print("Pickled!")


class BruteForce(POETForDMES):

    def __init__(self,
                 config: config.Config,
                 load_rewards=False,
                 save_models: bool = True,
                 **hyperparams):

        super().__init__(config,
                         optimiser_gradient=OptimiserGradientEnum.ES_Elite,
                         load_rewards=load_rewards,
                         save_models=save_models,
                         **hyperparams)

        self.verbose = False
        self.agents_simulated = self.agents

    def run(self, run_in_parallel=False):
        rewards = self.rewards
        print("\nStarting Brute Force...")

        for env in self.envs:
            if self.verbose: print(f"\tEnv {env}")
            for agent in self.agents:
                if self.verbose:  print(f"\t\tAgent {agent}", end="\t")
                r, i = self.reward(env, agent)
                if self.verbose: print(i)
                rewards.append((env, agent, r, i))

        self.df = pd.DataFrame(self.rewards, columns=self.DF_COLUMNS)
        self.save_rewards()
        print("Pickled!")
        print("Finished!")

        return rewards


class ReferenceApproach(POETForDMES):

    def __init__(self,
                 config: config.Config,
                 env_probabilities=None,
                 load_rewards=True,
                 **hyperparams):

        super().__init__(config,
                         load_rewards=load_rewards,
                         **hyperparams)
        
        self.verbose = False

        # self.envs = self.envs[:9]
        # for env in self.envs:
        #     print(f"Env {env}")
        #     for agent in self.agents:
        #         print(f"Agent {agent}")
        #         with open(utils.locator.get_model_results_path(
        #             env, agent, 0,
        #             ext='pickle'), 'rb') as file:
        #             data = pickle.loads(file.read())
        # get r2 and results
        # r,i = self.reward(env, agent)
        # self.data.append([env, agent, r, i])

        if load_rewards:
            self.load_bruteforce_results()

        if not env_probabilities:
            self.env_probabilities = [1]*len(self.envs)
    
    def load_bruteforce_results(self):
        path = utils.locator.get_rewards_path(
            self.project_path, BruteForce.__name__)
        try:
            with open(path, 'rb') as file:
                self.rewards = pickle.loads(file.read())
            self.df = pd.DataFrame(self.rewards, columns=self.DF_COLUMNS)
        except FileNotFoundError:
            print("Could not find file")
            self.rewards = []

    def run(self):
        # Assumes brute force has been run
        agents_top = []
        df = self.df
        rewards = []

        for env in self.envs:
            # R2=0 means not valid
            df_e = df[(df["Env"] == env) & (df["R2"] > 0)]
            a_idx = df_e["R2"].idxmin()
            agent_top = df["Agent"][a_idx]
            agents_top.append(agent_top)
            rewards.append((env, agent_top, df["Results"][a_idx], df["R2"][a_idx]))

        self.df = df[df["Agent"].isin(agents_top)]

        self.agents_simulated = agents_top
        self.rewards = rewards
        
        return self.rewards

    def plot(self, env, agent, env_to=None, agent_to=None, show=True):
        demand_path = os.path.join(
            utils.locator.get_profiles(self.project_path),
            'bld1.csv')

        flow_vis = vis.FlowNodeVisualiser(
            utils.locator.get_model_results_path(
                self.project_path,
                env,
                agent,
                num=1,
                ext='pickle'),
            env,
            agent,
            reset_demands=[[demand_path, 'elec']])
        flow_vis.plot(show=show)
        print()

        agent_compare_vis = vis.AgentPerformanceComparisonVisualiser(
            project_path=self.project_path,
            env_base=env, env_to=env_to,
            agent_base=agent, agent_to=agent_to,
            objectives=self.config.objectives,
        )

        agent_compare_vis.plot(show=show)


class RandomSample(POETForDMES):

    def __init__(self,
                 config: config.Config,
                 load_rewards=False,
                 save_models: bool = True,
                 **hyperparams):

        super().__init__(config,
                         optimiser_gradient=OptimiserGradientEnum.ES_Elite,
                         load_rewards=load_rewards,
                         save_models=save_models,
                         **hyperparams)

        self.verbose = False
        self.agents_simulated = self.agents
        self.num_samples = 700

    def run(self, run_in_parallel=False):
        rewards = self.rewards
        print("\nStarting Random Sample...")
        
        all_pairs = np.array(list(itertools.product(self.envs, self.agents)))
        sample_idx = np.random.randint(len(all_pairs), size=self.num_samples)
        self.pairs_sample = all_pairs[sample_idx]

        for env, agent in self.pairs_sample:
            if self.verbose: print(f"\tEnv {env}\tAgent {agent}", end="\t")
            r, i = self.reward(tuple(env), tuple(agent))
            if self.verbose: print(i)
            rewards.append((tuple(env), tuple(agent), r, i))

        self.df = pd.DataFrame(self.rewards, columns=self.DF_COLUMNS)
        self.save_rewards()
        print("Pickled!")
        print("Finished!")

        return rewards