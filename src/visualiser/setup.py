from dash_bootstrap_components._components.Card import Card
from ehub import Project
import os
from poet.poet import POETForDMES

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_table

import numpy as np
import pandas as pd

from utils import locator

# import main_uncertainty



PROJECT_PATH = os.path.join(os.path.dirname(os.path.dirname(
    __file__)), 'test_cases', 'boiler_chp')

ALGO_PICKLE_PATH = os.path.join(locator.get_outputs(PROJECT_PATH),
                            "poet_for_visualiser.pickle")

# POET
AGENT_PARAMS = [
    (1, 'chp_gas', 'bld1'),
    (1, 'boiler_gas', 'bld1'),
    # (1, 'pv', 'bld1'),
    # (1, 'battery', 'bld1'),
]
ENV_PARAMS = [
    ('elec_supply', 'bld1', 'costs_invest_per_use'),
    # ('elec_supply', 'bld1', 'co2_per_use'),
    ('gas_supply', 'bld1', 'costs_invest_per_use'),
    # ('gas_supply', 'bld1', 'co2_per_use'),
]

# Titles
ENV_TITLES = [
    "Electricity Price (CHF/kWh)",
    # "Electricity CO2e (kgCO2e/kWh)",
    "Gas Price (CHF/kWh)",
    # "Gas CO2e (kgCO2e/kWh)",
]
AGENT_TITLES = [
    "Gas CHP - Capacity (kW)",
    "Gas Boiler - Capacity (kW)",
    # "PV - Capacity (kW)",
    # "Battery - Capacity (kW)",
]
# finished_poet = main_uncertainty.run_poet(env_params=env_params, 
#                                           agent_params=agent_params,
#                                           from_pickle=LOAD_POET_RESULTS, 
#                                           debug=True)

import poet.solver as solver
from utils import locator
import dill

LOAD_POET_RESULTS = True

def load_algo():
    
    if LOAD_POET_RESULTS:
        assert os.path.isfile(ALGO_PICKLE_PATH)
        with open(ALGO_PICKLE_PATH, 'rb') as f:
            algo = dill.loads(f.read())
        algo.project_path = PROJECT_PATH
    else:
        algo = solver.setup(PROJECT_PATH,
                        ENV_PARAMS,
                        AGENT_PARAMS,
                        methodology=POETForDMES,
                        load_rewards=True,
                        verbose=False) 
        algo.load_rewards(locator.get_rewards_path(PROJECT_PATH, "BruteForcePOET"))
        algo.verbose = True
        algo = solver.solve(algo)
        
        with open(ALGO_PICKLE_PATH, 'wb') as f:
            dill.dump(algo, f)
    return algo


algo = load_algo()
algo.save_models = True
config = algo.config

# DATAFRAME

# Columns
ENV_KEYS = [str(e) for e in ENV_PARAMS]
env_1, env_2= ENV_KEYS # , env_3, env_4 
AGENT_KEYS = [str(a) for a in AGENT_PARAMS]
agent_1, agent_2 = AGENT_KEYS # , agent_3, agent_4
REWARD_COL = "reward" 
INDICATOR_COL = "indicator" 
INDICATOR_NORM_COL = "indicator_norm" 
ROBUSTNESS_COL = "robustness"

env_parent = 'env_parent'
agent_parent = 'agent_parent'
iteration_col = 'iteration'

columns = [*ENV_KEYS, *AGENT_KEYS, REWARD_COL, INDICATOR_COL]

# Indices
env_idx = slice(columns.index(env_1), 
                columns.index(env_2)+1)
agent_1_idx = np.arange(
    config._agent_lower[0], 
    config._agent_upper[0] + config._agent_step[0], 
    config._agent_step[0]
)
agent_2_idx = np.arange(
    config._agent_lower[1], 
    config._agent_upper[1] + config._agent_step[1], 
    config._agent_step[1]
)
agent_idx = slice(columns.index(env_1),
                  columns.index(env_2))

obj_1, obj_2 = algo.config.objectives
obj_1_idx = 0
obj_2_idx = 1




# DATAFRAME

df = pd.DataFrame([[*k[0], *k[1],  v[0], v[1]] 
                   for k,v in algo.reward_cache.items()],
                  columns=columns)

# df = pd.DataFrame.from_records(algo._logged_rewards[1:], columns=algo._logged_rewards[0])
# df[ENV_KEYS] = pd.DataFrame(df['env'].tolist(), index=df.index)
# df[AGENT_KEYS] = pd.DataFrame(df['agent'].tolist(), index=df.index)
# df = df[columns]

# Normalised indicators
df[INDICATOR_NORM_COL] = df[INDICATOR_COL] / \
    np.max(np.abs(df[INDICATOR_COL]), axis=0)

# Robustness
df[ROBUSTNESS_COL] = [np.nan] * len(df.index)

# Dataframe for active / archived pairs
pairs_active = np.array(algo.pairs_active)
df_active = pd.DataFrame(pairs_active.reshape(len(algo.pairs_active), 
                                              config.agent_size+config.env_size),
                         columns=[*ENV_KEYS, *AGENT_KEYS], dtype='float64')
if len(algo.pairs_archive) > 0:
    pairs_archive = np.array(algo.pairs_archive)
    df_archive = pd.DataFrame(pairs_archive.reshape(len(algo.pairs_archive), config.agent_size+config.env_size),
                              columns=[*ENV_KEYS, *AGENT_KEYS])
else:
    df_archive = pd.DataFrame()

# df.drop(columns=['env', 'agent'], axis=1, inplace=True)
df = df[[*ENV_KEYS, *AGENT_KEYS, 
         REWARD_COL, INDICATOR_COL, INDICATOR_NORM_COL,
        #  env_parent, agent_parent, iteration_col, 
         ROBUSTNESS_COL]]

# BOUNDS

# Bounds indicators
indicator_min = np.min(df[INDICATOR_COL])
indicator_max = np.max(df[INDICATOR_COL])

# Bounds rewards
rewards_min = []
rewards_max = []
for i, obj in enumerate(config.objectives):
    arr_flattened = np.array([r[:, i] for r in df[REWARD_COL]]).flatten()
    rewards_min.append(min(arr_flattened))
    rewards_max.append(max(arr_flattened))


AGENT_BOUNDS = [config._agent_lower[0], config._agent_upper[0]]
# agent_1_range = np.arange(start=config._agent_lower[0],
#                         stop=config._agent_upper[0],
#                         step=config._agent_step[0])

agent_2_bounds = [config._agent_lower[1], config._agent_upper[1]]
# agent_2_range = np.arange(start=config._agent_lower[1],
#                         stop=config._agent_upper[1],
#                         step=config._agent_step[1])


agent_range = [np.arange(start=config._agent_lower[i],
                        stop=config._agent_upper[i]+config._agent_step[i],
                        step=config._agent_step[i]) 
               for i in range(config.agent_size)]

env_range = [np.arange(start=config._env_lower[i],
                        stop=config._env_upper[i]+config._env_step[i],
                        step=config._env_step[i]) 
               for i in range(config.env_size)]

var_min = 0.0
var_max = 0.1
var_step = 0.01

# DEFAULTS
UPDATE_ON_HOVER = True
INTERPOLATE = True

# PARETO
pareto_x_idx = 0
pareto_y_idx = 1
pareto_layout = dict(
    plot_bgcolor='white',
    margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
    hovermode='closest',
    coloraxis_showscale=False,
    showlegend=False,
    uirevision=True
)

# DIVS
env_divs = []
for i in range(config.env_size):
    env_divs.append(dbc.Label(ENV_TITLES[i], className='h6'))
    env_divs.append(dcc.Slider(
                    id=f'env-{i+1}-slider',
                    min=config._env_lower[i],
                    max=config._env_upper[i],
                    value=config._env_default[i],
                    marks={step: str(step)
                           for step in env_range[i].round(decimals=2)},
                    step=config._env_step[i],
                    included=False))

agent_divs = []
for i in range(config.agent_size):
    agent_divs.append(dbc.Label(AGENT_TITLES[i], className='h6'))
    agent_divs.append(dcc.Slider(
        id=f'agent-{i+1}-slider',
        min=config._agent_lower[i],
        max=config._agent_upper[i],
        value=config._agent_default[i],
        marks={step: str(step) for step in agent_range[i].round(decimals=2)},
        step=config._agent_step[i],
        included=False))

ENV_DIVS = [
    dbc.Card([
        dbc.Label(f"Env Params", className='h4'),
        dbc.FormGroup(env_divs),
    ], body=True),
    dbc.Card([
        dcc.Graph(id='heatmap'),
    ])
]

AGENT_DIVS = [
    dbc.Card([
        dbc.Label(f"Agent Params", className='h4'),
        dbc.FormGroup(
            [ *agent_divs,
                ]),
    ], body=True),
    dcc.Graph(id='pareto'),
    dcc.Graph(id='flow_nodes'),
]

df_table = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

df_table[' index'] = range(1, len(df_table) + 1)

SETTINGS = [
    dbc.Card([
        dbc.Label("Settings", className='h5'),
        dbc.FormGroup(
            [
                dbc.Label(f"Env Settings", className='h6'),
                dcc.Checklist(
                    id='env-settings',
                    options=[dict(label=" Update on click", value="update"),
                             dict(label=" Interpolate", value="interpolate")],
                    value=["update", "interpolate"],
                    # labelStyle=dict(display='inline-block')
                ),
                dbc.Label(f"Agent Settings", className='h6'),
                dcc.Checklist(
                    id='agent-settings',
                    options=[dict(label=" Update on click", value="update"),
                             dict(label=" Lock zoom", value="lock-zoom")],
                    value=["update", "lock-zoom"],
                    labelStyle=dict(display='inline-block')),
                dbc.Button(
                    "Simulate",
                    id='simulate',
                    color="primary",
                ),
                dcc.Dropdown(
                    id='simulate-settings',
                    options=[
                        {'label': 'Fixed env, fixed agent', 'value': 'single'},
                        {'label': 'Fixed env, all agents', 'value': 'all-agents'},
                        {'label': 'All envs, fixed agent', 'value': 'all-envs'},
                    ],
                    value='single'
                ),
                dbc.Label(f"Variance", className='h6'),
                dcc.Slider(
                    id='agent-variance-slider',
                    min=var_min,
                    max=var_max,
                    value=0.0,
                    step=var_step,
                    included=True,
                    disabled=True,
                    vertical=True,
                ),
                # dbc.Label(f"Ranked Robust Agents", className='h6'),
                # dash_table.DataTable(
                #     columns=[
                #         {"name": i, "id": i} for i in ["Agent"]
                #     ],
                #     # row_selectable='single',
                #     id='agent_average_r2_table',
                #     # data=[0,2,3,4]
                # ),
                # dbc.Row([
                #     dbc.Col([
                #         dbc.Label(f"Average R2", className='h6'),
                #         dcc.Slider(
                #             id='agent-average-r2-slider',
                #             min=var_min,
                #             max=var_max,
                #             value=0.0,
                #             step=var_step,
                #             included=True,
                #             disabled=True,
                #             vertical=True,
                #         ),
                #     ], md=2),
                #     dbc.Col([
                #         dbc.Label(f"Variance", className='h6'),
                #         dcc.Slider(
                #             id='agent-variance-slider',
                #             min=var_min,
                #             max=var_max,
                #             value=0.0,
                #             step=var_step,
                #             included=True,
                #             disabled=True,
                #             vertical=True,
                #         ),
                #     ], md=2)  
                # ]),
            ]),
    ], body=True),
]


LAYOUT = dbc.Container(
    [
        # html.H1("DES with Uncertainty"),
        # html.Hr(),
        dbc.Row(
            [
                dbc.Col(SETTINGS, md=2),
                dbc.Col(ENV_DIVS, md=5),
                dbc.Col(AGENT_DIVS, md=5),
            ],
            align="top",
        )
    ],
    fluid=True,
)
