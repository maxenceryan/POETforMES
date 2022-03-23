class Config():
    """Contains the configuration for possible agent and environment instances.
    """

    def __init__(self,
                 network=None,
                 agent_keys=None,
                 env_keys=None,
                 component_capacity_keys=None,):

        self.agent_keys = agent_keys
        self.agent_map = []

        self.env_keys = env_keys
        self.env_map = []

        self.rewards_save_header = None

        if not component_capacity_keys:
            # see schema.toml
            # TODO add capacity / capacity_temp or capacity_fixed to indicate
            # which components have fixed capacity
            self._cap_keys = {
                'default': 'capacity_max',  # default as min
                'min': 'capacity_min',
                'max': 'capacity_max',
                'step': 'capacity_stepsize'
            }
        else:
            self._cap_keys = component_capacity_keys

        if network:
            self.network = network
            self._import_network(network)
            self._setup()
        else:
            raise NotImplementedError()

    def _import_network(self, network, cap_default='min'):
        """ Prepares the network to be the config for POET. 

            Finds out which parameters are fixed and which are variable
            Flatten mutable params for agent and mapping to place in the network config
            Flatten mutable params for env and mapping to place in the network config        
        """
        # assert all uncertain parameters mapped to either agent or env

        # setup agent
        self.agent = []  # default, min, max, stepsize
        for c in self.agent_keys:
            assert c in network.components.keys()
            c_rep = network.component_reps[c]
            _, c_name, c_loc = c
            component = getattr(network._project.components, c_name)
            c_cap_default = getattr(
                component, self._cap_keys['default']).value[c_loc]
            c_cap_min = getattr(component, self._cap_keys['min']).value[c_loc]
            c_cap_max = getattr(component, self._cap_keys['max']).value[c_loc]
            c_cap_step = getattr(
                component, self._cap_keys['step']).value[c_loc]

            # self.agent.append(
            #     [c_cap_default, c_cap_min, c_cap_max, c_cap_step])
            self.agent.append(
                [c_cap_min+c_cap_step, c_cap_min, c_cap_max, c_cap_step])
            self.agent_map.append(c_rep)

        # setup env
        self.env = []
        for c in self.env_keys:
            c_name, c_loc, c_attr = c
            component = getattr(network._project.components, c_name)
            attr_default, attr_min, attr_max, attr_step = getattr(
                component, c_attr).value[c_loc]

            self.env.append([attr_default, attr_min, attr_max, attr_step])
            self.env_map.append(getattr(component, c_attr))

        assert len(self.env_map) == len(self.env) and \
            len(self.agent_map) == len(self.agent)

        self.rewards_save_header = [*[(e.obj.name, e.name) for e in self.env_map],
                                    *[(*a.component_uid, 'capacity_max')
                                      for a in self.agent_map],
                                    'reward',
                                    'indicator']

    def _setup(self):
        self._agent_default, self._agent_lower, \
            self._agent_upper, self._agent_step = zip(*self.agent)
        self._env_default, self._env_lower, \
            self._env_upper, self._env_step = zip(*self.env)

        self.init_env = self._env_default
        self.init_agent = self._agent_default
        self.env_size = len(self._env_default)
        self.agent_size = len(self._agent_default)

        if self.is_network:
            self.objectives = self.network.project.scenario.objectives.value
            self.num_objectives = len(self.objectives)

    @property
    def is_network(self):
        return hasattr(self, 'network')
