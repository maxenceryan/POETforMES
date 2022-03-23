### FROM JACK HAWTHORNE MASTER THESIS
### added mutable capacity_min and capacity_max for modified POET


def set_up(network, model, objectives):
    pyomo_tools = network.project.pyomo_tools
    Param = pyomo_tools['Param']
    NNR = pyomo_tools['NonNegativeReals']

    ratios = {(id(k[0]), id(k[1])): v for k, v in network.ratios.items()}
    weights = {(id(k[0]), id(k[1]), k[2]): v for k, v in network.weights.items() if v != 1}

    model.ratios = Param(model.dependencies, initialize=ratios, domain=NNR)
    model.weights = Param(model.edges, model.part_loads, initialize=weights, domain=NNR, default=1)

    capacity_min = dict()
    capacity_max = dict()
    for node in network.nodes:
        mini, maxi = node.cap
        if mini != 0.0:
            capacity_min[id(node)] = mini
        if maxi != float('inf'):
            capacity_max[id(node)] = maxi

    model.capacity_min = Param(model.nodes, domain=NNR, initialize=capacity_min, mutable=True, default=0)
    model.capacity_max = Param(model.nodes, domain=NNR, initialize=capacity_max, mutable=True, default=100000000000000000000000000000000000000000000000000000000000000)

    # add the general component params
    component_params = dict()
    for component in network.project.components.instances():
        comp = getattr(network.project.components, component)
        for attr_name, attr in comp.attributes().items():

            co2_or_cost_param = bool(attr_name.startswith('co2') or attr_name.startswith('cost'))
            temporal_param = bool(attr_name.endswith('temp'))
            min_load_param = bool(attr_name == 'minimum_load')
            criteria_met = bool(co2_or_cost_param or temporal_param or min_load_param)
            
            if criteria_met and attr_name not in component_params:
                component_params[attr_name] = {}

            if temporal_param:
                start = network.project.scenario.period_start.value
                end = network.project.scenario.period_end.value 

                for location, periods in attr.value.items():
                    if (component, location) in model.components_general:
                        for period in range(start, end + 1):
                            component_params[attr_name][(comp.name, location, period)] = periods[period]

            elif co2_or_cost_param:
                for location, value in attr.value.items():
                    if (component, location) in model.components_general:
                        component_params[attr_name][(comp.name, location)] = value

            # elif min_load_param:
            #         fractions = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
            #         fractions = {f for f in fractions if f < attr.value}
            #         for location in comp.locations.value:
            #             if (component, location) in models[0].components_general:
            #                 for f in fractions:
            #                     component_params[attr_name][(comp.name, location, f)] = 0


    for attr, init in component_params.items():
        if attr.endswith('temp'):
            setattr(model, attr, Param(model.components_general, model.time, initialize=init, default=0, mutable=True))
        elif attr.startswith('cost') or attr.startswith('co2'):
            setattr(model, attr, Param(model.components_general, initialize=init, default=0, mutable=True))
        # elif attr == 'minimum_load':
        #     model.minimum_load = Param(model.components_general, model.part_loads, initialize=init, default=float('inf'))
    
    for obj in objectives:
        setattr(model, obj + '_upper_bound', Param(domain=NNR, mutable=True, default=100000000000000000000000000000000000000000000000000000000000000))
    
    return model