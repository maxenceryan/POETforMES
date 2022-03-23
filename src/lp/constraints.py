### FROM JACK HAWTHORNE MASTER THESIS

def set_up(network, model):
    pyomo_tools = network.project.pyomo_tools
    Constraint = pyomo_tools['Constraint']
    Set = pyomo_tools['Set']
    floor = pyomo_tools['floor']



    # def load_constraint(model, c_number, c_name, c_location, t):
    #     rep = model.component_reps[c_number, c_name, c_location][1]
    #     return model.loads[c_number, c_name, c_location, t] == floor((model.flow_node[rep, t]/model.capacity_max[rep]) * 10 + 0.5) * 10


    def general_edge_flow_limit(model, origin, destin, t):
        return model.flow_edge[origin, destin, t] <= model.flow_node[origin, t]


    def relative_edge_flow_limit(model, origin, destin, t):
        return model.flow_node[origin, t] == model.flow_edge[origin, destin, t]


    def area_edge_flow_limit(model, origin, destin, t): 
        return model.flow_node[origin, t] * model.capacity_max[origin] == model.flow_edge[origin, destin, t] 


    def store_edge_flow_limit(model, origin, destin, t):
        return model.flow_edge[origin, destin, t] <= model.flow_node[origin, t-1] * model.weights[origin, origin, 0]


    def from_nodes(model, node, t):
        return model.flow_node[node, t] == sum(model.flow_edge[edge, t] for edge in model.edges_from_node[node])


    def to_nodes(model, node, t):
        return model.flow_node[node, t] == sum(model.flow_edge[edge, t] * model.weights[edge, 0] for edge in model.edges_to_node[node])
     

    def stored(model, node, t):
        return model.flow_node[node, t] == \
            model.flow_node[node, t-1] * model.weights[node, node, 0] +\
            sum(model.flow_edge[edge, t] * model.weights[edge, 0] for edge in model.edges_to_node[node]) -\
            sum(model.flow_edge[edge, t] for edge in model.edges_from_node[node])
    

    def proportionality(model, dep, indep, t):
        return model.flow_node[dep, t] == model.ratios[dep, indep] * model.flow_node[indep, t]


    # def min_load_limit(model, node, t):
    #     component = model.n_to_c[node][1]
    #     rep = model.component_reps[component][1]
    #     return model.flow_node[node, t] <= model.minimum_load[component[1:3]] * model.flow_node


    def capacity_max_limit(model, node, t):
        return model.flow_node[node, t] <= model.capacity_max[node]


    def capacity_min_limit(model, node, t):
        return model.flow_node[node, t] >= model.capacity_min[node]


    def capacity_max_temp_limit(model, c_num, c_id, c_loc, t):
        component = (c_num, c_id, c_loc)
        rep = model.component_reps[component][1]
        return model.flow_node[rep, t] <= model.capacity_max_temp[c_id, c_loc, t]


    def capacity_min_temp_limit(model, c_num, c_id, c_loc, t):
        component = (c_num, c_id, c_loc)
        rep = model.component_reps[component][1]
        return model.flow_node[rep, t] >= model.capacity_min_temp[c_id, c_loc, t]

    
    def capacity_max_temp_solar_limit(model, c_num, c_id, c_loc, t):
        component = (c_num, c_id, c_loc)
        rep = model.component_reps[component][1]
        return model.flow_node[rep, t] == model.capacity_max_temp[c_id, c_loc, t]


    def capacity_min_temp_solar_limit(model, c_num, c_id, c_loc, t):
        component = (c_num, c_id, c_loc)
        rep = model.component_reps[component][1]
        return model.flow_node[rep, t] == model.capacity_min_temp[c_id, c_loc, t]


    comps = [x.component_uid for x in network.nodes_rep if x.node_type != 'solar']
    solar_comps = [x.component_uid for x in network.nodes_rep if x.node_type == 'solar']

    cap_max_temp_keys = {k[0:2] for k in model.capacity_max_temp.sparse_keys()}
    cap_min_temp_keys = {k[0:2] for k in model.capacity_min_temp.sparse_keys()}

    comps_max_temp = [c for c in comps if c[1:3] in cap_max_temp_keys]
    comps_min_temp = [c for c in comps if c[1:3] in cap_min_temp_keys]
    comps_max_temp_solar = [c for c in solar_comps if c[1:3] in cap_max_temp_keys]
    comps_min_temp_solar = [c for c in solar_comps if c[1:3] in cap_min_temp_keys]

    model.constrs_0 = Constraint(model.edges - model.edges_relative - model.edges_area - model.edges_store, model.time, rule=general_edge_flow_limit)
    model.constrs_1 = Constraint(model.edges_relative, model.time, rule=relative_edge_flow_limit)
    model.constrs_2 = Constraint(Set(initialize=comps_min_temp), model.time, rule=capacity_min_temp_limit)
    model.constrs_3 = Constraint(model.nodes_supply | (model.nodes_output - model.nodes_loose), model.time, rule=from_nodes)
    model.constrs_4 = Constraint(model.nodes_input|model.nodes_output|model.nodes_area|model.nodes_demand, model.time, rule=to_nodes)
    if model.nodes_store:
        model.constrs_6 = Constraint(model.edges_store, model.time, rule=store_edge_flow_limit)
        model.constrs_7 = Constraint(model.nodes_store, model.time, rule=stored)
    if model.nodes_area:
        model.constrs_8 = Constraint(model.edges_area, model.time, rule=area_edge_flow_limit)
    # if model.nodes_min_load:
    #     model.constrs_9 = Constraint(model.nodes_min_load, model.time, rule=min_load_limit)
    if model.dependencies:
        model.constrs_10 = Constraint(model.dependencies, model.time, rule=proportionality)
    if model.nodes_cap_max:
        model.constrs_11 = Constraint(model.nodes_cap_max - model.nodes_area, model.time, rule=capacity_max_limit)
    if model.nodes_cap_min:
        model.constrs_12 = Constraint(model.nodes_cap_min - model.nodes_area, model.time, rule=capacity_min_limit)
    if comps_max_temp:
        model.constrs_13 = Constraint(Set(initialize=comps_max_temp), model.time, rule=capacity_max_temp_limit)
    if comps_max_temp_solar:
        model.constrs_14 = Constraint(Set(initialize=comps_max_temp_solar), model.time, rule=capacity_max_temp_solar_limit)
    if comps_min_temp_solar:
        model.constrs_15 = Constraint(Set(initialize=comps_min_temp_solar), model.time, rule=capacity_min_temp_solar_limit)
    return model