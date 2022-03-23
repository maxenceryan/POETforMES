### FROM JACK HAWTHORNE MASTER THESIS

def set_up(network, model):
    pyomo_tools = network.project.pyomo_tools
    Set = pyomo_tools['Set']
    RangeSet = pyomo_tools['RangeSet']


    model.time = RangeSet(network.period_start, network.period_end)
    model.time_sub = RangeSet(network.period_start -1, network.period_end)
    model.part_loads = Set(initialize=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    for attr in network.__dict__:
        if attr == 'dependencies':
            deps = [(id(e[0]), id(e[1])) for e in getattr(network, attr)]
            setattr(model, attr, Set(initialize=deps))

        elif attr.startswith('edges'):
            edges = getattr(network, attr)
            edges_all = []
            edges_relative = []
            edges_area = []
            edges_store = []
            for e in edges:
                edge = [(id(e[0]), id(e[1]))]
                origin_type = e[0].node_type
                edges_all += [edge]
                if origin_type == 'input' or origin_type == 'solar':
                    edges_relative += [edge]
                elif origin_type == 'area':
                    edges_area += [edge]
                elif origin_type == 'store':
                    edges_store += [edge]
            setattr(model, 'edges', Set(initialize=edges_all))
            setattr(model, 'edges_relative', Set(initialize=edges_relative))
            setattr(model, 'edges_area', Set(initialize=edges_area))
            setattr(model, 'edges_store', Set(initialize=edges_store))

        elif attr == 'component_reps':
            c_reps = getattr(network, attr)
            comps = [c for c in c_reps]
            general_comps = [c[1:3] for c in comps]
            reps = {k: [id(v)] for k, v in c_reps.items()}
            setattr(model, 'components', Set(initialize=comps))
            setattr(model, 'components_general', Set(initialize=general_comps))
            setattr(model, attr, Set(model.components, initialize=reps))

        elif attr.startswith('nodes'):
            nodes = [id(n) for n in getattr(network, attr)]
            setattr(model, attr, Set(initialize=nodes))

    n_to_c = dict()
    nodes_cap_max = []
    nodes_cap_min = []
    nodes = []
    
    for n in network.nodes:
        node_id = id(n)
        mini, maxi = n.cap
        if mini != 0:
            nodes_cap_min += [node_id]
        if maxi != float('inf'):
            nodes_cap_max += [node_id]
        nodes += [node_id]
        n_to_c[node_id] = [n.component_uid]
    
    nodes_loose = []
    nodes_origin = set()
    nodes_destin = set()
    for edge in network.edges:
        nodes_origin.add(edge[0])
        nodes_destin.add(edge[1])
    nodes_loose = [id(n) for n in nodes_destin - nodes_origin if n.node_type == 'output']


    setattr(model, 'nodes', Set(initialize=nodes))
    setattr(model, 'n_to_c', Set(model.nodes, initialize=n_to_c))
    setattr(model, 'nodes_cap_max', Set(initialize=nodes_cap_max))
    setattr(model, 'nodes_cap_min', Set(initialize=nodes_cap_min))
    setattr(model, 'nodes_loose', Set(initialize=nodes_loose))

    # add indexed set of edges to and from nodes
    edges_to, edges_from = network.relations
    edges_to_node = dict()
    edges_from_node = dict()
    for n, edges in edges_to.items():
        edges_to_node[id(n)] = []
        for edge in edges:
            if edge[0].node_type != edge[1].node_type: # no store to store
                edges_to_node[id(n)] += [(id(edge[0]), id(edge[1]))]
    for n, edges in edges_from.items():
        edges_from_node[id(n)] = []
        for edge in edges:
            if edge[0].node_type != edge[1].node_type:
                edges_from_node[id(n)] += [(id(edge[0]), id(edge[1]))]

    setattr(model, 'edges_to_node', Set(model.nodes, initialize=edges_to_node))
    setattr(model, 'edges_from_node', Set(model.nodes, initialize=edges_from_node))
    return model