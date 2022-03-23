### FROM JACK HAWTHORNE MASTER THESIS
### Added cost an co2 bounds for epsilon constraint optimization

def set_up(network, model, objectives):
    pyomo_tools = network.project.pyomo_tools
    Constraint = pyomo_tools['Constraint']
    Objective = pyomo_tools['Objective']
    minimize = pyomo_tools['minimize']


    def energy_function(model):
        return model.energy == sum(
        [model.flow_node[node, t] for node in model.nodes_supply for t in model.time] +\
        [model.flow_node[store, model.time[1]-1] for store in model.nodes_store] +\
        [0])


    def cost_function(model):
        m = model
        return m.cost == sum(
            sum(
            [m.costs_invest_fixed[m.n_to_c[rep][1][1:3]]] +\
            [m.costs_OM_fixed[m.n_to_c[rep][1][1:3]]] +\
            [(m.costs_invest_per_cap[m.n_to_c[rep][1][1:3]] + m.costs_OM_per_cap[m.n_to_c[rep][1][1:3]]) * m.capacity_max[rep]] +\
            [(
                m.costs_invest_per_use_temp[m.n_to_c[rep][1][1:3], t] +\
                m.costs_OM_per_use_temp[m.n_to_c[rep][1][1:3], t] +\
                m.costs_invest_per_use[m.n_to_c[rep][1][1:3]] +\
                m.costs_OM_per_use[m.n_to_c[rep][1][1:3]]
                ) * m.flow_node[rep, t] \
            for t in m.time]) for rep in m.nodes_rep)


    def co2_function(model):
        m = model
        return m.co2 == sum(sum(
            [m.co2_fixed[m.n_to_c[rep][1][1:3]]] + \
            [m.co2_per_cap[m.n_to_c[rep][1][1:3]]] + \
            [(m.co2_per_use_temp[m.n_to_c[rep][1][1:3], t] + m.co2_per_use[m.n_to_c[rep][1][1:3]]) * m.flow_node[rep, t] for t in m.time] +\
            [0]) for rep in m.nodes_rep) # put this in so you can deselect whichever by line and it wont crash


    def cost_bound(model):
        return model.cost <= model.cost_upper_bound

    def co2_bound(model):
        return model.co2 <= model.co2_upper_bound
    
    def energy_expr(model):
        return model.energy


    functions = {
        'energy': energy_function,
        'cost': cost_function,
        'co2': co2_function
    }
    
    bounds = {
        # 'energy': energy_bound,
        'cost': cost_bound,
        'co2': co2_bound
    }

    for obj in objectives:
        setattr(model, 'objective_function_' + obj, Constraint(rule=functions[obj]))
        setattr(model, 'objective_bound_' + obj, Constraint(rule=bounds[obj]))
        setattr(model, obj + '_objective', Objective(expr=getattr(model, obj), sense=minimize))
        getattr(model, obj + '_objective').deactivate()

    # for i, obj in enumerate(objectives):
    #     setattr(models[i], 'objective_' + obj, Objective(expr=getattr(models[i], obj), sense=minimize))
    # for i, obj in enumerate(objectives):
    #     other_objs = [o for o in objectives if o != obj]
    #     for o in other_objs:
    #         getattr(models[i], o+'_objective').deactivate()
    return model