### FROM JACK HAWTHORNE MASTER THESIS

import graphviz
import os
from . import locator

def visualise(network, file_name):

    dot = graphviz.Digraph('network')
    dot.format = 'png'
    dot.attr(rankdir='LR')
    dot.attr('node', shape='point')

    network_nodes = set(node.no_cap for node in network.nodes)
    network_edges = set((edge[0].no_cap, edge[1].no_cap) for edge in network.edges)

    nodes_origin = set()
    nodes_destin = set()
    nodes_demand = set(node.no_cap for node in network.nodes_demand)
    nodes_source = set(node.no_cap for node in set().union(network.nodes_supply, network.nodes_solar))

    for edge in network_edges:
        nodes_origin.add(edge[0])
        nodes_destin.add(edge[1])

    nodes_demand = nodes_demand.intersection(nodes_destin - nodes_origin)
    nodes_source = nodes_source.intersection(nodes_origin - nodes_destin)

    with dot.subgraph(name='cluster_demand') as c:
        c.attr(style='invis')
        for node in nodes_demand:
            c.node(str(node))

    with dot.subgraph(name='cluster_supply') as c:
        c.attr(style='invis')
        for node in nodes_source:
            c.node(str(node))

    for edge in network_edges:
        color = 'lightgrey'
        if edge[0][0:3] == edge[1][0:3]:
            style = 'dashed'
        else:
            style = 'solid'
        dot.edge(str(edge[0]), str(edge[1]), color=color, style=style)


    path = os.path.join(locator.get_outputs(network.project.project_path), file_name)
    dot.render(path, view=False)

if __name__ == "__main__":
    pass