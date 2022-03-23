import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (RBF, ConstantKernel,
                                                     DotProduct,
                                                     ExpSineSquared, Matern,
                                                     RationalQuadratic)

import visualiser.setup as setup
from poet.solver import COLUMNS_AVERAGE_R2
from visualiser.visualiser import FlowNodeVisualiser

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server

agent_1, agent_2= setup.agent_1, setup.agent_2#, setup.agent_3, setup.agent_4
env_1, env_2 = setup.env_1, setup.env_2#, setup.env_3, setup.env_4
indicator_col = setup.INDICATOR_COL
reward_col = setup.REWARD_COL
indicator_norm_col = setup.INDICATOR_NORM_COL
robustness_col = setup.ROBUSTNESS_COL

df = setup.df

def is_env(env, negate=False):
    idx = [0, 1]
    keys = setup.ENV_KEYS
    defaults = setup.config._env_default
    return _generate_encoding_query(env, idx, keys, defaults, negate=negate)

def is_agent(agent, negate=False):
    idx = [0, 1]
    keys = setup.AGENT_KEYS
    defaults = setup.config._agent_default
    return _generate_encoding_query(agent, idx, keys, defaults, negate=negate)
    
def _generate_encoding_query(encoding, idx, keys, defaults, negate=False):
    # a1, a2 = agent
    # query_setup = f"`{agent_1}`=={agent[0]} & `{agent_2}`=={agent[1]}"
    query = []
    for i, k in enumerate(keys):
        if i in idx:
            query.append(f'`{k}`=={float(encoding[i])}')
        else:
            query.append(f'`{k}`=={float(defaults[i])}')
    result = ' & '.join(query)
    return result if not negate else f'~({result})'
    # return (df[agent_1] == agent[0]) & (df[agent_2] == agent[1])

finished_poet = setup.algo

app.layout = setup.LAYOUT

# CALLBACKS
simulate_n_clicks = 0

# UPDATE HEATMAP
# @functools.lru_cache(maxsize=32)
@app.callback(
    Output('heatmap', 'figure'),
    # Output('heatmap2', 'figure'),
    [Input('env-settings', 'value'),
     Input('simulate', 'n_clicks'),
     Input('simulate-settings', 'value'),
    *[Input(f'env-{i+1}-slider', 'value') for i in range(setup.config.env_size)],
    *[Input(f'agent-{i+1}-slider', 'value') for i in range(setup.config.agent_size)],
    #  Input('env-1-slider', 'value'),
    #  Input('env-2-slider', 'value'),
    #  Input('agent-1-slider', 'value'),
    #  Input('agent-2-slider', 'value')
     ])
def update_agent_heatmap(env_settings, simulate_clicks, simulate_settings,
                         env_1_value, env_2_value,
                         agent_1_value, agent_2_value):
    
    interpolate, simulate_on_click = 'interpolate' in env_settings, 'update' in env_settings    
    if env_1_value == 0: env_1_value = 0.2
    elif env_1_value == 10: env_1_value = 10.2
    elif env_1_value == 20: env_1_value = 20.2
    
    if env_2_value == 0: env_2_value = 0.1
    elif env_2_value == 1: env_2_value = 1.1
    elif env_2_value == 2: env_2_value = 2.1
    # env = args[:setup.config.env_size]
    # agent = args[setup.config.env_size:]
    env = (env_1_value, env_2_value)
    agent = (agent_1_value, agent_2_value)
    
    print("heatmap: ",env,agent)
    
    
    simulate(simulate_on_click, simulate_clicks, simulate_settings,
             env, agent)
    
    x_idx, y_idx = 0, 1
    # figs = []
    # for x,y in [(agent_1, agent_2), (agent_3, agent_4)]:
    fig = create_heatmap(interpolate, env, agent_1, agent_2)
    fig.update_layout(
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        # background='white',
        coloraxis_showscale=False,
        showlegend=False,
        xaxis=dict(title=setup.AGENT_TITLES[x_idx], range=setup.AGENT_BOUNDS),
        yaxis=dict(title=setup.AGENT_TITLES[y_idx], range=setup.agent_2_bounds))
    # figs.append(fig)
        # TODO fix the xmin, xmax, ymin, ymax
    return fig

def simulate(simulate_on_click, simulate_clicks, simulate_settings,
             env, agent):
    global simulate_n_clicks, df
    
    def reset_range(reward):
        x_idx = 0
        y_idx = 1
        
        reward_x = reward[:,x_idx]
        reward_y = reward[:,y_idx]
        
        if max(reward_x) > setup.rewards_max[x_idx]:
            setup.rewards_max[x_idx] = max(reward_x)
        if min(reward_x) > setup.rewards_min[x_idx]:
            setup.rewards_min[x_idx] = min(reward_x)
        if max(reward_y) > setup.rewards_max[y_idx]:
            setup.rewards_max[y_idx] = max(reward_y)
        if min(reward_y) > setup.rewards_min[y_idx]:
            setup.rewards_min[y_idx] = min(reward_y)
    
    def not_simulated():
        return (df[reward_col] == None)
    
    if None in locals().values():
        return
    
    if simulate_on_click \
        and simulate_clicks > simulate_n_clicks:
        # try to find env agent pair
        
        if simulate_settings == 'single':
            dff = df.query(is_agent(agent) + ' & ' + is_env(env))
        elif simulate_settings == 'all-envs':
            dff = df.query(is_agent(agent))
        elif simulate_settings == 'all-agents':
            dff = df.query(is_env(env))
        else:
            dff = df
        
        dff = dff[not_simulated()]

        # if not yet simulated
        if len(dff.index) == 0:
            # finished_poet.setup_reward_model()
            reward, indicator = finished_poet.reward(env, agent)
            
            if indicator:
                indicator_norm = indicator / setup.indicator_max
                reset_range(reward)
            else:
                print("Indicator is null, no solution found")
                indicator_norm = None
            # if indicator > setup.indicator_max:
            #     df[indicator_col] *= setup.indicator_max/np.abs(indicator)
            # else:
            # add reward to df
            df = df.append(pd.Series([*env, *agent, 
                                    reward, indicator, indicator_norm, None
                                    # None, None, None, None
                                    ], index=df.columns), 
                            ignore_index=True)
            # df.loc[df.index[-1]+1] = [*env, *agent, 
            #                           reward, indicator, indicator_norm, 
            #                           None, None, None, None]
            # robustness = update_robustness(env, agent)
        
        simulate_n_clicks += 1

def get_rewards_range():
    x = dict(title="cost", linecolor='black',
            range=[setup.rewards_min[setup.pareto_x_idx]*0.9, 
                setup.rewards_max[setup.pareto_x_idx]*1.1])
    y = dict(title="co2", linecolor='black',
            range=[setup.rewards_min[setup.pareto_y_idx]*0.9, 
                    setup.rewards_max[setup.pareto_y_idx]*1.1])
    return x,y

def update_robustness(env, agent):
    dff = df.query(is_agent(agent))
    r = dff[indicator_norm_col].std()
    df[robustness_col].mask(is_agent(agent), r, inplace=True)
    return r

def interpolate_gaussian(dff, Z):
    X = setup.agent_2_idx
    Y = setup.agent_1_idx
    gp_y = np.array(dff[indicator_norm_col], dtype='float')
    np.where(~np.isnan(gp_y))
    gp_X = np.stack([dff[agent_1], dff[agent_2]], axis=1)
    
    # kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) 
    #     # + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
    
    # gp = GaussianProcessRegressor(kernel = kernel,
    #                               normalize_y=True,
    #                               alpha=0.1,
    #                             #   n_restarts_optimizer=10,
    #                               )
    
    # param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
    #           "kernel": [ExpSineSquared(l, p)
    #                      for l in np.logspace(-2, 2, 10)
    #                      for p in np.logspace(0, 2, 10)]}
    # gp = GridSearchCV(KernelRidge(), param_grid=param_grid) 
    
    # scaler = StandardScaler()
    # scaler.fit(gp_X)  # Don't cheat - fit only on training data
    # gp_X = scaler.transform(gp_X)
    # X_test = scaler.transform(X_test)
    # gp =  SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=0.1)

    # print(gp.kernel.get_params())
    
    gp = make_pipeline(PolynomialFeatures(6), Ridge())
    gp.fit(gp_X, gp_y)
    
    Xgrid, Ygrid = np.meshgrid(X, Y)
    all_X = np.stack([Xgrid.ravel(), Ygrid.ravel()], axis=1)
    # gp_y_samples = gp.sample_y(all_X, n_samples=10)
    gp_y_samples = gp.predict(all_X)
    gp_Z = gp_y_samples.reshape(len(Y), len(X))
    Z = np.array(Z, dtype = 'float64')
    Z = np.where(np.isnan(Z), gp_Z, Z)
    return Z

def create_heatmap(interpolate, env, x_key, y_key):
    dff = df.query(is_env(env))
    if len(dff) > 0:
        X_obs = dff[x_key]
        Y_obs = dff[y_key]
        Z_obs = dff[indicator_norm_col]
        
        # query = ' & '.join([f'`{k}`=={v}' for k, v in setup.AGENT_KEYS)
        # dff = dff.query(is_agent(agent))
        dff = dff.drop_duplicates([x_key, y_key])
        df_pivot = dff[[x_key, y_key, indicator_norm_col]].pivot(
            index=x_key, columns=y_key, values=indicator_norm_col)
        df_pivot = df_pivot.reindex(
            setup.agent_1_idx, 
            columns=setup.agent_2_idx)
        X = setup.agent_2_idx
        Y = setup.agent_1_idx
        Z = df_pivot.values
        if interpolate:
            Z = interpolate_gaussian(dff, Z)
        fig = px.imshow(Z, x=X, y=Y,
                        # zmin=range_indicator[0], zmax=range_indicator[1],
                        origin='lower')
        
        dff_active = pd.merge(setup.df_active, dff, how='inner', 
                          on=[env_1, env_2, x_key,y_key])
        # plot the active points
        X_active, Y_active = dff_active[agent_1], dff_active[agent_2]
        fig.add_trace(go.Scatter(x=X_active, y=Y_active, xaxis='x1', name='active_pair',
                                    mode='markers', 
                                    marker=dict(color='rgba(135, 206, 250, 0)',size=15,
                                        line=dict(width=2,
                                            color='white')),
                                    customdata=Z_obs, hoverinfo='skip'))
        
        if interpolate:
            # fig.add_trace(go.Image(z=Z_obs, xaxis=X_obs, yaxis=Y_obs))
            fig.add_trace(go.Scatter(x=Y_obs, y=X_obs, xaxis='x1', name='simulated_pairs',
                                    mode='markers', marker_color='white',
                                    customdata=Z_obs, hoverinfo='skip'))
            
        # # plot the optimising attempts
        # fig.add_shape(type="line",
        #     x0=dff[agent_1][0], y0=dff[agent_2][0], 
        #     x1=1, y1=2,
        #     line=dict(
        #         color="LightSeaGreen",
        #     )
        # )
    else:
        fig = go.Figure()
    return fig

# UPDATE PARETO
@app.callback(
    [Output('pareto', 'figure'),
     Output('agent-variance-slider', 'value')],
    [Input('agent-settings', 'value'),
     Input('agent-1-slider', 'value'),
     Input('agent-2-slider', 'value'),
    #  *[Input(f'env-{i+1}-slider', 'value') for i in range(setup.config.env_size)],
    #  *[Input(f'agent-{i+1}-slider', 'value') for i in range(setup.config.agent_size)],
     Input('env-1-slider', 'value'),
     Input('env-2-slider', 'value')
     ],
    State('pareto', 'relayoutData'))  # prevent_initial_call=True
def update_agent_pareto(agent_settings, 
                        agent_1_value, agent_2_value,
                        env_1_value, env_2_value,
                        relayout_data=True):
  
    lock_zoom = 'lock-zoom' in agent_settings
    
    # env = args[:setup.config.env_size]
    # agent = args[setup.config.env_size:
    #     setup.config.env_size+setup.config.agent_size]
    # relayout_data = args[-1]
    if env_1_value == 0: env_1_value = 0.2
    elif env_1_value == 10: env_1_value = 10.2
    elif env_1_value == 20: env_1_value = 20.2
    
    if env_2_value == 0: env_2_value = 0.1
    elif env_2_value == 1: env_2_value = 1.1
    elif env_2_value == 2: env_2_value = 2.1
    
    
    env = (env_1_value, env_2_value)
    agent = (agent_1_value, agent_2_value)
    print(env, agent)
    
    try:
        fig = create_agent_pareto(env, agent)
    except Exception as ex:
        print(ex)
        fig = go.Figure()
    fig.update_layout(**setup.pareto_layout)
    
    if relayout_data and lock_zoom == 'Lock View':
        if 'xaxis.range[0]' in relayout_data:
            
            fig.layout.xaxis.range = [
                relayout_data['xaxis.range[0]'],
                relayout_data['xaxis.range[1]']
            ]
        if 'yaxis.range[0]' in relayout_data:
            fig.layout.yaxis.range  = [
                relayout_data['yaxis.range[0]'],
                relayout_data['yaxis.range[1]']
            ]
    else:
        x_range, y_range = get_rewards_range()
        fig.update_xaxes(x_range)
        fig.update_yaxes(y_range)
    
    robustness = get_robustness(env, agent)
    
    return fig, robustness

def get_robustness(env, agent):
    r = df.query(is_env(env) + ' & ' + is_agent(agent))[robustness_col]
    if len(r) == 0:
        return setup.var_min
    r = list(r)[0]
    if r is None:
        return setup.var_min
    elif r >= setup.var_min and r <= setup.var_max:
        return round(r, ndigits=2)
    elif r > setup.var_max:
        return setup.var_max
    else:
        return setup.var_min

def create_agent_pareto(env, agent,
                        show_other_envs=True,
                        show_other_agents=False):
    
    dff_agent = df.query(is_agent(agent))
    dff_other_envs = dff_agent.query(is_env(env, negate=True))
    dff_other_envs = dff_other_envs[setup.columns]
    reward_selected = dff_agent.query(is_env(env))[reward_col]

    if reward_selected.first_valid_index() is None:
        fig = go.Figure()
    else:
        X_selected, Y_selected = np.stack(reward_selected[reward_selected.first_valid_index()], axis=1)
        fig = go.Figure(go.Scatter(x=X_selected, y=Y_selected,
                                    name=f"Env: {env}",
                                    marker=dict(color='red'),
                                    customdata=[env]*len(X_selected),
                                    ))
        
        if show_other_envs:
            rewards_other = dff_other_envs[reward_col]
            if len(rewards_other) > 0:
                reward_idx = dff_other_envs.columns.get_loc(reward_col)
                # env_idx = dff_other_envs.columns.get_loc(setup.env_idx)
                
                for entry in dff_other_envs.values:
                    env_other = entry[setup.env_idx]
                    X_other_envs, Y_other_envs = np.stack(entry[reward_idx], axis=1)
                    fig.add_trace(go.Scatter(x=X_other_envs, y=Y_other_envs,
                                            name=f"Env: {env_other}",
                                            marker=dict(color='black'),
                                            customdata=[env_other]*len(X_selected),
                                            ))

        if show_other_agents:
            pass
        
        # fig.update_traces(customdata=[dff[env_1], dff[env_2]])
        fig.update_traces(mode='markers+lines',
                            hovertemplate=f"{setup.obj_1}: %{{x:,.0f}} CHF<br>" +
                                            f"{setup.obj_2}: %{{y:,.0f}} kgCO2e")
    return fig

# import pickle
# import dill
# from utils import locator

# def create_agent_comparer(env_base, env_to,
#                           agent_base, agent_to):
    
#     def import_data(reset_demands=None):
#         path_base = locator.get_model_results_path(
#             setup.PROJECT_PATH,
#             env_base,
#             agent_base,
#             num,
#             ext='pickle')

#         path_to = locator.get_model_results_path(
#             setup.PROJECT_PATH,
#             self.env_to,
#             self.agent_to,
#             self.num,
#             ext='pickle')

#         path_instance_base = locator.get_model_instance_path(
#             setup.PROJECT_PATH,
#             self.env_base,
#             self.agent_base,
#             self.num)

#         path_instance_to = locator.get_model_instance_path(
#             self.project_path,
#             self.env_to,
#             self.agent_to,
#             self.num)

#         with open(path_base, 'rb') as file:
#             self.data_base = pickle.loads(file.read())

#         with open(path_to, 'rb') as file:
#             self.data_to = pickle.loads(file.read())

#         with open(path_instance_base, 'rb') as file:
#             self.instance_base = dill.loads(file.read())

#         with open(path_instance_to, 'rb') as file:
#             self.instance_to = dill.loads(file.read())

#     diffs = dict()  # key=component, value=difference
#     base = {k: v for k, v in data_base.items() if ast.literal_eval(k)[
#         3]}
#     to = {k: v for k, v in data_to.items() if ast.literal_eval(k)[3]}

#     base_cost, base_co2, base_energy = ()
#     to_cost, to_co2, to_energy = ()

#     diffs = {
#         'energy': {
#             'battery': -100.0,
#             'pv': 20,
#             'boiler_gas': 50
#         },
#         'cost': {
#             'battery': -100.0,
#             'pv': 20,
#             'boiler_gas': 50
#         }
#     }

#     fig = go.Figure()
#     for k, v in diffs.items():
#         fig.add_bar(x=list(v.keys()),
#                     y=list(v.values()),
#                     name=str(k))
#     return fig

# @app.callback(
#     Output('agent_average_r2_table', 'data'),
#     [Input('agent-1-slider', 'value'),
#      Input('agent-2-slider', 'value'),
#      Input('env-1-slider', 'value'),
#      Input('env-2-slider', 'value')
#      ],
#     # State('pareto', 'relayoutData')
#     )  # prevent_initial_call=True
# def update_average_r2_table(agent_1_value, agent_2_value,
#                             env_1_value, env_2_value):
    
#     dff = df[[agent_1, agent_2, indicator_col]].groupby([agent_1, agent_2]).mean()
#     rows = {"Agent" : str(a) for a in dff.sort_values(by=indicator_col)[:10].index}
#     return rows


@app.callback(
    Output('flow_nodes', 'figure'),
    Input('pareto', 'clickData'),
    [State('agent-settings', 'value'),
     *[State(f'env-{i+1}-slider', 'value') for i in range(setup.config.env_size)],
     *[State(f'agent-{i+1}-slider', 'value') for i in range(setup.config.agent_size)],
    ])  # prevent_initial_call=True
def create_flow_node_plot(clickData, agent_settings, *args):
    env = args[:setup.config.env_size]
    agent = args[setup.config.env_size:]
    
    try:
        num = clickData['points'][0]['pointNumber']
        if num > 0: num = 1
        
        vis = FlowNodeVisualiser(setup.PROJECT_PATH, env, agent, num)
        fig = vis.plot(show=False)
        fig.update_layout(plot_bgcolor='white',
                            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                            hovermode='closest',
                            coloraxis_showscale=False,
                            showlegend=False,
                            uirevision=True)
        # print(fig['data'])
    except:
        fig = go.Figure()
    return fig

# UPDATE AGENT PARAMS
@app.callback(
    # [Output('agent-1-slider', 'value'),
    #  Output('agent-2-slider', 'value')],
    [Output(f'agent-{i+1}-slider', 'value') for i in range(setup.config.agent_size)],
    
    Input('heatmap', 'clickData'),
    [State('agent-settings', 'value'),
     *[State(f'agent-{i+1}-slider', 'value') for i in range(setup.config.agent_size)],
    #  State('agent-1-slider', 'value'),
    #  State('agent-2-slider', 'value')
     ])
def update_agent(clickData, agent_settings, *agent):
    update = 'update' in agent_settings
    if clickData and len(clickData['points']) > 0:
        point = clickData['points'][0]
        if update \
                and (('z' in point and (point['z'] is None or ~np.isnan(point['z'])))
                     or 'customdata' in point):
            return float(point['y']), float(point['x'])
                    # *agent[2:])
    return [float(a) for a in agent]

# UPDATE ENV PARAMS
@app.callback(
    # [Output('env-1-slider', 'value'),
    #  Output('env-2-slider', 'value')],
    [Output(f'env-{i+1}-slider', 'value') for i in range(setup.config.env_size)],
    Input('pareto', 'clickData'),
    #  Input('agent_average_r2_table', 'clickData')],
    [State('env-settings', 'value'),
    #  *[State(f'env-{i+1}-slider', 'value') for i in range(setup.config.env_size)],
     State('env-1-slider', 'value'),
     State('env-2-slider', 'value')
     ])
def update_env(clickData,  env_settings, env_1_value, env_2_value):
    if env_1_value == 0: env_1_value = 0.2
    elif env_1_value == 10: env_1_value = 10.2
    elif env_1_value == 20: env_1_value = 20.2
    
    if env_2_value == 0: env_2_value = 0.1
    elif env_2_value == 1: env_2_value = 1.1
    elif env_2_value == 2: env_2_value = 2.1
    # env = args[:setup.config.env_size]
    # agent = args[setup.config.env_size:]
    env = (env_1_value, env_2_value)
    
    # env_1_value, env_2_value = env
    try:
        update = 'update' in env_settings
        if update:
            print(clickData['points'][0]['customdata'])
            env_1_value, env_2_value = clickData['points'][0]['customdata']
    finally:
        return env_1_value, env_2_value
   
def update_marks(col_target, col_match, col_match_value, precision=None):        
    if precision == 'int':
        key_func = int
        value_func = int
    elif precision == 'float':
        key_func = lambda x : x
        value_func = lambda x : "{:.2f}".format(x)
    else:
        key_func = int
        value_func = int
    return {key_func(i) : str(value_func(i))
        for i in set(df[[col_target, col_match]]
                        [df[col_match] == col_match_value]
                        [col_target]
                        .values)}

# UPDATE ENV MARKERS
@app.callback(
    [Output('env-1-slider', 'marks'),
    Output('env-2-slider', 'marks')],
    # [Output(f'env-{i+1}-slider', 'marks') for i in range(setup.config.env_size)],
    
    [Input('pareto', 'clickData'),
    #  Input('agent_average_r2_table', 'clickData'),
    #  *[Input(f'env-{i+1}-slider', 'value') for i in range(setup.config.env_size)],
     Input('env-1-slider', 'value'),
     Input('env-2-slider', 'value')
     ], # TODO listen to simulate clicks
    State('env-settings', 'value')
    )
def update_env_marks(clickData, env_1_value, env_2_value, env_settings=None):
    if env_1_value == 0: env_1_value = 0.2
    elif env_1_value == 10: env_1_value = 10.2
    elif env_1_value == 20: env_1_value = 20.2
    
    
    if env_2_value == 0: env_2_value = 0.1
    elif env_2_value == 1: env_2_value = 1.1
    elif env_2_value == 2: env_2_value = 2.1
    try:
        update = 'update' in env_settings
        if update:
            env_1_value, env_2_value = clickData['points'][0]['customdata']
    finally:
        env_1_value, env_2_value = float(env_1_value), float(env_2_value)
        env_1_marks = update_marks(env_1, env_2, env_2_value, precision='float')
        env_2_marks = update_marks(env_2, env_1, env_1_value, precision='float')
        return env_1_marks, env_2_marks

# UPDATE AGENT MARKERS
@app.callback(
    [Output('agent-1-slider', 'marks'),
    Output('agent-2-slider', 'marks')],
    # [Output(f'agent-{i+1}-slider', 'marks') for i in range(setup.config.agent_size)],
    [Input('heatmap', 'clickData'),
    #   *[Input(f'agent-{i+1}-slider', 'value') for i in range(setup.config.agent_size)],
     Input('agent-1-slider', 'value'),
     Input('agent-2-slider', 'value')
     ], # TODO listen to simulate clicks
    State('agent-settings', 'value')
    )
def update_agent_marks(clickData, agent_1_value, agent_2_value, agent_settings=None):
    try:
        update = 'update' in agent_settings
        if update:
            agent_1_value, agent_2_value  = clickData['points'][0]['customdata']
    finally:
        agent_1_value, agent_2_value = float(agent_1_value), float(agent_2_value) 
        agent_1_marks = update_marks(agent_1, agent_2, agent_2_value, precision='int')
        agent_2_marks = update_marks(agent_2, agent_1, agent_1_value, precision='int')
        return agent_1_marks, agent_2_marks #, None, None


if __name__ == '__main__':
    app.run_server(debug=True)