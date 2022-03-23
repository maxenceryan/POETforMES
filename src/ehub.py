### EXTENDED FROM JACK HAWTHORNE MASTER THESIS
### Added functionality to set the network and some cosmetic fixes

from utils.validation import Mode
from utils import parse
from utils import validation
from utils import visualiser
import lp.solver
import os
import json
import csv
import random
from pyomo.environ import (ConcreteModel, Set,
    Var, Param, Constraint, Reals, NonNegativeReals,
    RangeSet, Objective, minimize, maximize, floor)

VERBOSE = True

def __get_name__(within, find):
    for key, value in within.__dict__.items():
        if value is find:
            return key


class Project:
    def __init__(self, project_path=None):
        self.profiles = dict()
        self.scenario = None
        self.project_path = project_path

    @property
    def pyomo_tools(self):
        return {'ConcreteModel':ConcreteModel, 'Set':Set, 'Var': Var, 'Param':Param, 
                'Constraint': Constraint, 'Reals':Reals, 'NonNegativeReals':NonNegativeReals,
                'RangeSet':RangeSet, 'Objective':Objective, 'minimize':minimize, 
                'maximise': maximize, 'floor':floor}

    @property
    def solvers(self):
        return {'lp': lp.solver}

      
    @property
    def network(self):
        return Network(self)

    @property
    def scenario(self):
        if self._scenario is None:
            return ValueError("no scenario selected")
        return self._scenario

    @scenario.setter
    def scenario(self, arg):
        if arg is None:
            self._scenario = arg
            return
        assert(arg in self.scenarios.instances()), "scenario of name '%s' not found" %arg
        self._scenario = getattr(self.scenarios, arg)

    @property
    def project_path(self):
        return self._project_path

    @project_path.setter
    def project_path(self, arg):
        if arg is None:
            self._project_path = arg
        elif os.path.isdir(arg):
            self._project_path = arg
            self.load()
        else:
            raise FileNotFoundError("%s"%arg)


    def load(self, path=None):
        if path is None:
            path = self.project_path
        config = {k:v for k, v in parse.parse_config(path).items() if k in validation.schema}
        # create all libraries
        for lib in validation.schema:
            setattr(self, lib, Library(self))
        # fill in all the instances for each library
        for obj, instances in config.items():
            lib = getattr(self, obj)
            for instance in instances:
                setattr(lib, instance, eHub_Object(lib))
        # validate the input
        for lib, objs in config.items():
            for obj, attrs in objs.items():
                self.add(lib, obj, attrs)
        # parse all the profiles
        for file_path, args in self.profiles.items():
            with open(file_path, 'r') as f:
                data = list(csv.reader(f, delimiter=','))
            for arg in args:
                self.profiles[file_path][arg].update({i+1:v for i, v in enumerate(parse.parse_profile(data, *arg))})
        # set the scenario to the first
        assert(self.scenarios), "no scenarios present"
        self.scenario = self.scenarios.instances()[0]


    def dump(self):
        filename = 'config'
        incl = ['co2', 'cost', 'capacity']
        data = {}
        for group in incl:
            data.update({attr_name: attr.value for attr_name, attr in getattr(self, group).items()})
        with open(os.path.join(self.project_path, 'config.json'),'w') as f:
            f.write(json.dumps(data, indent=4))
            print(f"config written to {filename}.json")
        # return


    def run(self, arg):
        assert(self.scenario), "the scenario has not been set"
        assert(arg in self.solvers)
        self.solvers[arg]


    def add(self, lib, obj_name, kwargs={}):
        if VERBOSE: print("adding '%s' to %s library" %(obj_name, lib))
        lib = getattr(self, lib)
        if not hasattr(lib, obj_name):
            setattr(lib, obj_name, eHub_Object(lib))
        obj = getattr(lib, obj_name)
        # delete unknown attributes
        for attr in set(kwargs) - set(validation.schema[lib.name]):
            del kwargs[attr]
        # get defaults for missing attributes
        for attr in set(validation.schema[lib.name]) - set(kwargs):
            if 'default' in validation.schema[lib.name][attr]:
                kwargs.update({attr:validation.schema[lib.name][attr]['default']})
            else:
                print(attr) # should be error here

        for attr_name in validation.schema[lib.name]:
            if attr_name in obj.__dict__:
                attr = getattr(obj, attr_name)
                setattr(attr, "value", kwargs[attr_name])


class Library:
    def __init__(self, project):
        self.project = project

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, arg):
        if isinstance(arg, Project):
            self._project = arg
        else:
            raise ValueError("Library only accepts '%s'" %Project)

    @property
    def name(self):
        return __get_name__(within=self.project, find=self)

    def instances(self):
        return [k for k in self.__dict__ if k not in ['name', '_project']]

    
    def pprint(self):
        r = "\nLibrary: \n\t" + str(self.name) + "\nObjects:\n"
        for i in self.instances():
            r += "\t"+str(i) + "\n"
        print(r)


class eHub_Object:
    def __init__(self, lib):
        self.lib = lib
        for attr in validation.schema[self.lib.name]:
            setattr(self, attr, Attribute(self.lib.project, self))      

    @property
    def lib(self):
        return self._lib

    @lib.setter
    def lib(self, arg):
        if isinstance(arg, Library):
            self._lib = arg
        else:
            raise ValueError("eHub_Object only accepts '%s'" %Library)

    @property
    def name(self):
        return __get_name__(within=self.lib, find=self)

    def attributes(self):
        return {k:v for k,v in self.__dict__.items() if k not in ['name', '_lib']}

    def pprint(self):
        r = "\nLibrary:\n\t" + str(self.lib.name) +"\nObject:\n\t" + str(self.name) + "\nAttributes:\n"
        for attr_name, attr in self.attributes().items():
            r += "\t" + attr_name + ': ' + attr.__repr__() + "\n"
        print(r)


class Attribute:
    def __init__(self, project, obj, value=None):
        self.project = project
        self.obj = obj
        if value is None:
            pass
        else:
            self.value = value
            
        if validation.mode == Mode.BOUNDED:
            pass
            

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, arg):
        if isinstance(arg, Project):
            self._project = arg.__getattribute__
        else:
            raise ValueError("eHub_Object only accepts '%s'" %Project)

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, arg):
        if isinstance(arg, eHub_Object):
            self._obj = arg
        else:
            raise ValueError("eHub_Object only accepts '%s'" %eHub_Object)

    @property
    def name(self):
        return __get_name__(within=self.obj, find=self)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, arg):
        self._value = validation.validate(self.project, self.obj, self.name, arg, validation.schema[self.obj.lib.name][self.name])


    def info(self):
        print(validation.schema[self.obj.lib.name][self.name])
        return

    def __repr__(self):
        r = str()
        if isinstance(self.value, type(dict())):
            r += "{"
            for k, v in self.value.items():
                if hasattr(v, "__iter__"):
                    if len(v)> 10:
                        v = str(v).split(',')
                        v = ",".join([v[0]]+[' ... ']+[v[-1]])
                v = str(v)
                r += "'%s': "%k + v + ", "
            r += "}"
        else:
            r += str(self.value)
        return "Attribute" + r

    # def __iter__(self):
    #     def flatten(data, keys=[]):            
    #         if isinstance(data, dict):
    #             return reduce(list.__add__, [flatten(v,keys+[k]) for k, v in data.items()], [])
    #         else:
    #             return [tuple(keys)+ (data,)]
    #     for tup in flatten(self.value):
    #         if tup[0:-1]:
    #             yield (self.obj.name,) + tup[0:-1], tup[-1]
    #         else:
    #             yield None, tup[-1]


class Network:
    def __init__(self, project):
        self.project = project
        self.nodes_area = set()
        self.nodes_solar = set()
        self.nodes_input = set()
        self.nodes_output = set()
        self.nodes_supply = set()
        self.nodes_demand = set()
        self.nodes_store = set()
        self.nodes_rep = set()
        self.nodes_min_load = set()
        self.component_reps = dict()
        self.edges = set() # tuple two nodes
        self.weights = dict() # index = edge
        self.dependencies = set() # tuple two nodes
        self.ratios = dict()
   
    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, arg):
        if isinstance(arg, Project):
            self._project = arg
        else:
            raise ValueError("network only accepts '%s'" %Project)

    @property
    def period_start(self):
        return self.project.scenario.period_start.value
    
    @property
    def period_end(self):
        return self.project.scenario.period_end.value
    
    @property
    def new(self):
        return Network

    @property
    def nodes(self):
        return set().union(*(getattr(self, k) for k in self.__dict__ if k.startswith('nodes')))

    @property
    def components(self):
        components = {}
        for node in self.nodes:
            if node.component_uid not in components:
                components[node.component_uid] = set()
            components[node.component_uid].add(node)
        return components

    @property
    def relations(self):
        nodes = self.nodes
        edges_to = {n:[] for n in nodes}
        edges_from = {n:[] for n in nodes}
        for edge in self.edges:
            edges_to[edge[1]] += [edge]
            edges_from[edge[0]] += [edge]
        return edges_to, edges_from

    @property
    def super_config(self):
        network = Network(self.project)
        for component in self.project.components.instances():
            for location in getattr(self.project.components, component).locations.value:
                contin = True
                while contin:
                    contin = self.add_component(component, location, network)
        all_sources = set().union(network.nodes_output, network.nodes_supply, network.nodes_solar)
        all_sinks = set().union(network.nodes_input, network.nodes_demand, network.nodes_area)
        for location in self.project.locations.instances():
            for energy_carrier in self.project.energy_carriers.instances():
                sources = [s for s in all_sources if s.node_location == location and s.node_energy_carrier == energy_carrier]
                sinks = [s for s in all_sinks if s.node_location == location and s.node_energy_carrier == energy_carrier]
                for so in sources:
                    for si in sinks:
                        if so.component_uid != si.component_uid:
                            network.edges.add((so, si))
        return network


    def add_component(self, component_name, location, network=None):
        if network is None:
            network = self
        comp = getattr(self.project.components, component_name)
        number_max = comp.number_max.value[location]
        count = 1
        the_count = self.count(network)
        if (component_name, location) in the_count:
            count = the_count[(component_name, location)] + 1
        del the_count
        if count > number_max:
            return False

        component_uid = (count, component_name, location)

        # here we translate and incorporate the nodes
        translate = {}
        change_cap_min = dict()
        change_cap_max = dict()
        for node, props in comp.nodes.value.items():
            node_location = location
            if 'location' in props:
                node_location = (props['location'])
            rep_status = bool(node == 'rep')
            capacity_min = 0.0
            capacity_max = float('inf')
            if rep_status and  props['node_type'] != 'demand':
                if not comp.capacities.value[location]:
                    choices = [comp.capacity_min.value[location]]
                    x = int((comp.capacity_max.value[location]-comp.capacity_min.value[location])/comp.capacity_stepsize.value[location])
                    for _ in range(x):
                        choices += [choices[-1]+comp.capacity_stepsize.value[location]]
                    if choices[0] == 0.0:
                        choices.pop(0)
                    comp.capacities.value[location] = choices
                capacity_max = random.choice(comp.capacities.value[location])
            if 'capacity_min' in props:
                change_cap_min[node] = props['capacity_min']
            if 'capacity_max' in props:
                change_cap_max[node] = props['capacity_max']
            n = Node(component_uid[0], component_uid[1], component_uid[2], rep_status,
                            node_location, props['energy_carrier'], props['node_type'], capacity_min, capacity_max)
            translate[node] = n
            getattr(network, 'nodes_' + translate[node].node_type).add(n)

        for node in change_cap_min:    
            translate[node].capacity_min = change_cap_min[node] * translate['rep'].node_capacity_max
        for node in change_cap_max:
            translate[node].capacity_max = change_cap_max[node] * translate['rep'].node_capacity_max

        network.nodes_rep.add(translate['rep'])
        network.component_reps[component_uid] = translate['rep']
        del change_cap_min
        del change_cap_max

        # here we translate and incorporate the ratios
        for dep, indeps in comp.ratios.value.items():
            for indep, ratio in indeps.items():
                dependency = (translate[dep], translate[indep])
                network.dependencies.add(dependency)
                network.ratios[dependency] = ratio

        # here we translate and incorporate the weights, adding in any missing ones
        all_fractions = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
        for origin, destins in comp.weights.value.items():
            for destin, fractions in destins.items():
                edge = (translate[origin], translate[destin])
                f_defined = sorted([int(f) for f in fractions])
                f_undefined = sorted(all_fractions - set(f_defined))
                for a in f_undefined:
                    less_than = [b for b in f_defined if a>b]
                    if not less_than:
                        fractions[str(a)] = 0.0
                    else:
                        fractions[str(a)] = fractions[str(max(less_than))]
                for fraction, weight in fractions.items():
                    network.edges.add(edge)
                    network.weights[edge + (int(fraction),)] = weight

        if comp.minimum_load.value > 0:
            network.nodes_min_load = network.nodes_min_load.union(translate.values())
        return True
    
    def add_component_edge(self, origin, destination):
        """Adds possible edge between two components"""
        o = self.components[origin]
        d = self.components[destination]
        for o_node in o:
            for d_node in d:
                if o_node.node_energy_carrier == d_node.node_energy_carrier \
                    and d_node.node_type in validation.NODES_ALLOWED_INTERCONNECTIONS[o_node.node_type]:
                    # TODO is checking location necessary?
                    self.add_edge(o_node, d_node)

    def add_edge(self, origin, destin):
        """Adds edge between two nodes"""
        components = self.components
        if origin.component_uid in components:
            for node in components[origin.component_uid]:
                if node.no_cap == origin.no_cap:
                    origin = node
                
        if destin.component_uid in components:
            for node in components[destin.component_uid]:
                if node.no_cap == destin.no_cap:
                    destin = node
        self.edges.add((origin, destin))


    def count(self, network=None):
        if network is None:
            network = self
        component_instances = set(node.component_uid for node in network.nodes_rep)
        component_IDs = list(comp[1:3] for comp in component_instances)
        component_instances = list(comp[1:3] for comp in component_instances)
        return {tuple(ID): component_instances.count(ID) for ID in component_IDs}


    def view(self, name=None):
        if name is None:
            name = str(id(self))
        return visualiser.visualise(self, name)

    def show(self):
        print('Network: %s' %id(self))
        for k in sorted(self.__dict__):
            print('\t', end='')
            print(k, self.__dict__[k])
        print('\n')
            
    
    def network_to_tuple(self, network=None):
        if network is None:
            network = self
        return tuple(sorted([(tuple(edge[0]), tuple(edge[1])) for edge in network.edges]))
    
    def set_network_config(self, network_config: dict):
        # Validate the network_config
        if VERBOSE: print("Setting network config...")
        validation.validate_network_config(self.project, network_config)
        print("Network validated.")

        for location, components_dict in network_config['components'].items():
            for component_name, number in components_dict.items():
                for i in range(number):
                    self.add_component(component_name, location)
        for location, components_dict in network_config['connections'].items():
            for origin, destinations in components_dict.items():
                for d in destinations:
                    # TODO hardcoded number....
                    origin_uid  = (1, origin, location)
                    destination_uid  = (1, d, location)
                    self.add_component_edge(origin_uid, destination_uid)
        print(f"Network set.")

class Node:
    def __init__(self, component_number, component_ID, component_location,
                    rep_status, node_location, node_energy_carrier, node_type,
                    node_capacity_min, node_capacity_max):

        self.component_number = component_number
        self.component_ID = component_ID
        self.component_location = component_location
        self.rep_status = rep_status
        self.node_location = node_location
        self.node_energy_carrier = node_energy_carrier
        self.node_type = node_type
        self.node_capacity_min = node_capacity_min
        self.node_capacity_max = node_capacity_max

    @property
    def component_uid(self):
        return (self.component_number, self.component_ID, self.component_location)

    @property
    def no_cap(self):
        return (self.component_number, self.component_ID, self.component_location,
                self.rep_status, self.node_location, self.node_energy_carrier, self.node_type)
    @property
    def cap(self):
        return (self.node_capacity_min, self.node_capacity_max)
    
    def __repr__(self):
        return "Node" + str(tuple(self))


    def __iter__(self):
        order = [self.component_number, self.component_ID, self.component_location,
                self.rep_status, self.node_location, self.node_energy_carrier, self.node_type,
                self.node_capacity_min, self.node_capacity_max]
        for x in order:
            yield x
