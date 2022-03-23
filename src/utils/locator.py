### EXTENDED FROM JACK HAWTHORNE MASTER THESIS

import os
from pathlib import Path


def get_inputs(project_path):
    return os.path.join(project_path, 'inputs')


def get_outputs(project_path):
    return os.path.join(project_path, 'outputs')


def get_config(project_path):
    return os.path.join(get_inputs(project_path), 'config')


def get_config_files(project_path):
    return list(Path(get_config(project_path)).glob('**/*.toml'))


def get_csv_config(project_path):
    return os.path.join(get_inputs(project_path), 'csv_config')


def get_profiles(project_path):
    return os.path.join(get_inputs(project_path), 'profiles')


def get_schema():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'schema', 'schema.toml')


def get_csv_schema():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'schema', 'csv_schema.yaml')


def get_template():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'template', 'ehub_data.yaml')


def get_tests():
    return list(Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')).glob('**/*.json'))


def get_poet_hyperparameters(project_path):
    return os.path.join(get_inputs(project_path), 'poet_hyperparameters.toml')


def get_fixed_network(project_path):
    return os.path.join(get_inputs(project_path), 'network.toml')


def get_rewards_path(project_path, method, ext='pickle'):
    return os.path.join(get_outputs(project_path), f'rewards_{method}.{ext}')


def get_rewards_csv(project_path):
    return os.path.join(get_outputs(project_path), 'rewards.csv')


def get_model_dir(project_path, env, agent):
    env_str = "-".join(str(e) for e in env)
    agent_str = "-".join(str(a) for a in agent)

    return os.path.join(get_outputs(project_path), 'models',
                        f'model_E-{env_str}_A-{agent_str}')


def get_model_results_path(project_path, env, agent, num, ext="txt"):
    return os.path.join(get_model_dir(project_path, env, agent), f'model_results_{num}.{ext}')


def get_model_instance_path(project_path, env, agent, num):
    return os.path.join(get_model_dir(project_path, env, agent), f'model_instance_{num}.pickle')


def get_optimality_path(project_path, ext="csv", num=0):
    return os.path.join(get_outputs(project_path), f"optimality_scores_{num}.{ext}")

def get_optimality_ranked_path(project_path, ext="csv", num=0):
    return os.path.join(get_outputs(project_path), f"optimality_scores_ranked_{num}.{ext}")

def create_outputs_dir(project_path):
    path = os.path.join(get_outputs(project_path), 'models')
    if not os.path.isdir(path):
        os.mkdir(path)
        
def create_model_dir(project_path, env, agent):
    path = get_model_dir(project_path, env, agent)
    if not os.path.isdir(path):
        os.mkdir(path)




if __name__ == "__main__":
    pass
