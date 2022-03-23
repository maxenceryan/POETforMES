import ast
import csv
import pickle

import dill
import plotly.graph_objects as go
import toml
import utils.locator
import utils.parse
import pyomo.core.kernel as pyo


class Visualiser:

    def __init__(self):
        pass

    def import_data(self):
        pass

    def setup(self):
        pass

    def plot(self):
        pass

    # def energy_function(model, nodes):
    #     return {
    #         n: sum(
    #             [model.flow_node[node, t] for node in model.nodes_supply for t in model.time] +
    #             [model.flow_node[store, model.time[1]-1] for store in model.nodes_store] +
    #             [0])
    #         for n in nodes
    #     }

    def cost_function(model):
        m = model
        return {
            m.n_to_c[rep][1][1:3]: pyo.value(
                sum(
                    [m.costs_invest_fixed[m.n_to_c[rep][1][1:3]]] +
                    [m.costs_OM_fixed[m.n_to_c[rep][1][1:3]]] +
                    [(m.costs_invest_per_cap[m.n_to_c[rep][1][1:3]] + m.costs_OM_per_cap[m.n_to_c[rep][1][1:3]]) * m.capacity_max[rep]] +
                    [(
                        m.costs_invest_per_use_temp[m.n_to_c[rep][1][1:3], t] +
                        m.costs_OM_per_use_temp[m.n_to_c[rep][1][1:3], t] +
                        m.costs_invest_per_use[m.n_to_c[rep][1][1:3]] +
                        m.costs_OM_per_use[m.n_to_c[rep][1][1:3]]
                    ) * m.flow_node[rep, t]
                        for t in m.time]))
            for rep in m.nodes_rep
        }

    def co2_function(model):
        m = model
        return {
            m.n_to_c[rep][1][1:3]: pyo.value(
                sum(
                    [m.co2_fixed[m.n_to_c[rep][1][1:3]]] +
                    [m.co2_per_cap[m.n_to_c[rep][1][1:3]]] +
                    [(
                        m.co2_per_use_temp[m.n_to_c[rep][1][1:3], t] + 
                        m.co2_per_use[m.n_to_c[rep][1][1:3]]
                    ) * m.flow_node[rep, t] 
                    for t in m.time
                    ] +
                [0]))
            for rep in m.nodes_rep
        }


class FlowNodeVisualiser(Visualiser):

    def __init__(self, project_path: str,
                 env: tuple, agent: tuple,
                 num: int = 0,
                 chart_type='scatter',
                 reset_demands=None):
        self.path = utils.locator.get_model_results_path(project_path, 
                                                         env, agent, 
                                                         num, ext="pickle")
        self.env = env
        self.agent = agent
        self.chart_type = chart_type
        self.import_data(reset_demands=reset_demands)
        self.setup()

    def import_data(self, reset_demands=None):
        ext = self.path.split('.')[-1]
        if ext == 'pickle':
            with open(self.path, 'rb') as file:
                self.data = pickle.loads(file.read())
        elif ext == 'toml':
            with open(self.path, 'r') as file:
                self.data = toml.loads(file.read())
        else:
            raise NotImplementedError

        # reset demands
        if reset_demands:
            for path, column in reset_demands:
                with open(path, 'r') as f:
                    profile = list(csv.reader(f, delimiter=','))
                profile_parsed = utils.parse.parse_profile(profile, column)
                for entry in self.data:
                    if entry.find('demand') >= 0 and entry.find(column) >= 0:
                        self.data[entry] = profile_parsed
                        break

    def setup(self):
        self.timerange = len(list(self.data.values())[1])

    def plot(self, show=True) -> go.Figure():
        fig = go.Figure()
        x = list(range(self.timerange+1))
        for k, v in self.data.items():
            if self.chart_type == 'scatter':
                k_tuple = ast.literal_eval(k)
                fig.add_scatter(x=x,
                                y=v,
                                name=f"{k_tuple[1]}<br>{k_tuple[5]}<br>{k_tuple[6]}",
                                mode='markers+lines',
                                customdata=[self.env]*len(x))
                                # hovertemplate= = "%{text}<br>(%{k[1]:.2f}, %{k[5]:.2f}, %{k[6]:.2f})",)
            elif self.chart_type == 'bar':
                fig.add_bar(x=x,
                            y=v,
                            name=str(k))

        if self.chart_type == 'bar':
            fig.update_layout(barmode='stack',)
        
        if show:
            fig.show()

        return fig


class AgentPerformanceComparisonVisualiser(Visualiser):

    def __init__(self, project_path: str,
                 env_base=None, env_to=None,
                 agent_base=None, agent_to=None,
                 objectives=None, num=None):
        self.project_path = project_path
        self.num = num

        assert (env_base and env_to) \
            or (agent_base and agent_to), "Beed to compare either envs or agents"
        self.env_base = env_base
        self.env_to = env_to
        self.agent_base = agent_base
        self.agent_to = agent_to

        self.data = dict()
        self.import_data()
        self.setup()

        self.comparers = ['energy'] + objectives

    def import_data(self, reset_demands=None):

        path_base = utils.locator.get_model_results_path(
            self.project_path,
            self.env_base,
            self.agent_base,
            self.num,
            ext='pickle')

        path_to = utils.locator.get_model_results_path(
            self.project_path,
            self.env_to,
            self.agent_to,
            self.num,
            ext='pickle')

        path_instance_base = utils.locator.get_model_instance_path(
            self.project_path,
            self.env_base,
            self.agent_base,
            self.num)

        path_instance_to = utils.locator.get_model_instance_path(
            self.project_path,
            self.env_to,
            self.agent_to,
            self.num)

        with open(path_base, 'rb') as file:
            self.data_base = pickle.loads(file.read())

        with open(path_to, 'rb') as file:
            self.data_to = pickle.loads(file.read())

        with open(path_instance_base, 'rb') as file:
            self.instance_base = dill.loads(file.read())

        with open(path_instance_to, 'rb') as file:
            self.instance_to = dill.loads(file.read())

    def setup(self):
        self.diffs = dict()  # key=component, value=difference
        base = {k: v for k, v in self.data_base.items() if ast.literal_eval(k)[
            3]}
        to = {k: v for k, v in self.data_to.items() if ast.literal_eval(k)[3]}

        base_cost, base_co2, base_energy = ()
        to_cost, to_co2, to_energy = ()

        self.diffs = {
            'energy': {
                'battery': -100.0,
                'pv': 20,
                'boiler_gas': 50
            },
            'cost': {
                'battery': -100.0,
                'pv': 20,
                'boiler_gas': 50
            }
        }

    def plot(self, show=True) -> go.Figure():
        fig = go.Figure()
        for k, v in self.diffs.items():
            fig.add_bar(x=list(v.keys()),
                        y=list(v.values()),
                        name=str(k))

        if show:
            fig.show()

        return fig
