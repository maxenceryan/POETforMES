### FROM JACK HAWTHORNE MASTER THESIS

def set_up(network, model, objectives):
    pyomo_tools = network.project.pyomo_tools
    Var = pyomo_tools['Var']
    Param = pyomo_tools['Param']
    NNR = pyomo_tools['NonNegativeReals']
    Reals = pyomo_tools['Reals']
    
    before_start = network.period_start - 1

    model.flow_edge = Var(model.edges, model.time, domain=NNR)
    model.flow_node = Var(model.nodes, model.time_sub, domain=NNR)
    
    for obj in objectives:
        if obj == 'cost':
            setattr(model, obj, Var(domain=Reals))
        else:
            setattr(model, obj, Var(domain=NNR))
    return model