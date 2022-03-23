### EXTENDED FROM JACK HAWTHORNE MASTER THESIS
### Added functionality for handling uncertain parameters and a network definition file

from . import parse
from . import locator
from functools import reduce
import os
from enum import Enum

schema = parse.parse_schema()

# NODE_TYPES = schema['components']['nodes']['keys']['rep']['keys']['node_type']['enum']

# inter aka between components, directional aka in form origin: destinations
NODES_ALLOWED_INTERCONNECTIONS = {
    'supply': ['demand', 'input'],
    'demand': [],
    'input': [],
    'output': ['demand', 'input'],
    'store': ['demand', 'input'],
    'area': ['input'], # TODO
    'solar': ['area'] # TODO
}

# intra aka within components
# NODES_ALLOWED_INTRACONNECTIONS = {
#     'supply': [],
#     'demand': [],
#     'input': ['output'],
#     'output': [],
#     'store': [],
#     'area': [],
#     'solar': []
# }

class Mode(Enum):
    DETERMINISTIC = 0
    BOUNDED = 1

mode = Mode.DETERMINISTIC
bounded = []

def validate(project, obj, attr, value, schema):


    def check_profile(project, obj, attr, value, schema):
        if not value:
            return list(value)
        file_name = value[0]
        args = value[1:]
        profile_path = os.path.join(locator.get_profiles(project("project_path")), file_name)
        assert(os.path.isfile(profile_path)), "'%s' is not a valid file path" %profile_path
        if profile_path not in project('profiles'):
            project('profiles')[profile_path] = {}
        if tuple(args) not in project('profiles')[profile_path]:
            project('profiles')[profile_path].update({tuple(args): dict()})
        return project('profiles')[profile_path][tuple(args)]
        

    def check_dict(project, obj, attr, value, schema):
        assert('keys' in schema), "schema for dict type attribute: '%s' does not have keys" %attr
        schema = schema['keys']
        unique_keys = set(schema) - {'ref',  'self_ref', 'optional', 'value', 'enum'}
        missing_keys = unique_keys - set(value)
        if 'optional' in schema:
            missing_keys = missing_keys - set(schema['optional'])
        assert(not missing_keys - set(value.keys())), "the attribute '%s' from '%s' is missing '%s'" %(attr, obj.name, ', '.join(missing_keys))
        new_value = {}
        for k, v in value.items():
            if 'ref' in schema or 'self_ref' in schema:
                for x in check_ref(project, obj, k, schema):
                    if x not in value:
                        new_value[x] = v
                    else:
                        new_value[x] = value[x]
            elif 'enum' in schema:
                new_value[check_enum(project, obj, attr, k, schema)] = v
            else:
                new_value[k] = v
        for k, v in new_value.items():
            if k in unique_keys:
                new_value[k] = validate(project, obj, attr, v, schema[k])
            elif 'value' not in schema:
                raise KeyError("attribute '%s' of '%s' is not recognised as a generic key" %(k, obj.name))
            else:
                new_value[k] = validate(project, obj, attr, v, schema['value'])
        return new_value


    def check_ref(project, obj, value, schema):
        ref = list()
        if 'ref' in schema:
            reference = schema['ref']
            for library in reference:
                ref += project(library).instances()
        elif 'self_ref' in schema:
            reference = schema['self_ref']
            for attribute in reference:
                ref += list(getattr(obj, attribute).value)
        if value in ref:
            return [value]
        elif value == 'all':
            return ref
        else:
            raise ValueError("'%s' not found in '%s': %s"%(value, obj.name, ref))


    def check_enum(project, obj, attr, value, schema):
        if value not in schema['enum']:
            raise ValueError("the value: '%s' for attribute '%s' not a valid. \n\nPlease choose from:  %s." %(
                    value, attr,  ', '.join([str(x) for x in schema['enum']])))
        return value


    def check_list(project, obj, attr, value, schema):
        assert('item' in schema), "schema for attribute: '%s' does not have any item declaration" %attr
        schema = schema['item']
        if 'ref' in schema or 'self_ref' in schema:
            value = [check_ref(project, obj, v, schema) for v in value]
            return list(set(reduce(lambda i, j: i+j, value)))
        return [validate(project, obj, attr, v, schema) for v in value]
    

    def check_reg(project, obj, attr, value, schema):
        
        def check_bounded(value):
            """ bounds must be of form (default, lower, upper, step) """
            # check length
            if not schema['dtype'] in ['int', 'float']:
                return value
            assert isinstance(value, list) and len(value) == 4, f"'{obj}:{attr}' - Bounds must be in the form [default, upper, lower, step]"
            try:
                value = [reg_functions[schema['dtype']](v) for v in value]
            except TypeError:
                raise TypeError(f"'{obj}:{attr}' - {value} does not match the required type {schema['dtype']}.")
            
            default, lower, upper, step = value
            assert default <= upper and default >= lower
            # check step is divisible considering floating point errors
            assert round(upper - default, ndigits=4) % step < 0.0001 and round(default - upper, ndigits=4) % step < 0.0001, f"'{obj}:{attr}' - Ranges [{lower}, {default}] and [{default}, {upper}] are not divisible by step {step}."
            for loc in obj.locations.value:
                bounded.append((obj.name, loc, attr))
            return value
        
        reg_functions = {
            'int': int,
            'str': str,
            'float': float,
            'bool': bool
        }
        
        if 'ref' in schema or 'self_ref' in schema:
            value = check_ref(project, obj, value, schema)[0]
        
        if mode == Mode.DETERMINISTIC:
            try:
                value = reg_functions[schema['dtype']](value)
            except ValueError:
                raise ValueError("'%s:%s' requires type '%s'" %(obj, attr, schema['dtype']))
            return reg_functions[schema['dtype']](value)
        
        elif mode == Mode.BOUNDED:
            try:
                value = reg_functions[schema['dtype']](value)
            except TypeError:
                try:
                    value = check_bounded(value)
                except TypeError:
                    raise TypeError("'%s:%s' requires bounded type '%s'" %(obj, attr, schema['dtype']))
            return value

    functions = {
        'int': check_reg,
        'str': check_reg,
        'float': check_reg,
        'bool': check_reg,
        'list': check_list,
        'dict': check_dict,
        'enum': check_enum,
        'profile': check_profile
    }
    
    return functions[schema['dtype']](project, obj, attr, value, schema)

    # validation.validate(self.project, self.obj.name, self.name, arg, validation.schema[self.obj.lib.name][self.name])
    
def validate_network_config(project, network_config):
    
    nodes_matched = {}
    
    def nodes_can_match(origin, destination):
        """"Checks if which nodes match between the origin and destination components.
        Updates the nodes_matched dictionary."""
        result = False
        for o_node, o_node_props in origin.nodes.value.items():
            for d_node, d_node_props in destination.nodes.value.items():
                if o_node_props['energy_carrier'] == d_node_props['energy_carrier'] \
                    and d_node_props['node_type'] in NODES_ALLOWED_INTERCONNECTIONS[o_node_props['node_type']]:
                    nodes_matched.update({
                        (origin.name, o_node): True,
                        (destination.name, d_node): True,
                    })
                    result = True
        return result
     
    assert 'components' in network_config
    assert 'connections' in network_config
    
    # check components
    for location, component_dict in network_config['components'].items():
        for component_name, component_number in component_dict.items():
            assert component_name in project.components.instances(), f"Component '{component_name}' is not in the project."
            component = getattr(project.components, component_name)
            assert location in component.locations.value, f"Location '{location}' is not in the component '{component.name}'."
            number_max = component.number_max.value[location]
            number_min = component.number_min.value[location]
            assert number_min <= component_number <= number_max, \
                    f"""Component number is {component_number} but 
                        should be >= {number_min} and <= {number_max}."""
            if component.name == 'boiler_gas':
                print()
            [nodes_matched.update({(component_name, node): False}) for node in component.nodes.value.keys()]


    # check connections
    for location, connection_dict in network_config['connections'].items():
        assert location in project.locations.instances(), f"Location '{location}' is not in the project."
        for origin, destinations in connection_dict.items():
            components = project.components.instances()
            assert origin in components, f"Component '{origin}' is not in the project."
            origin_component = getattr(project.components, origin)
            for destination in destinations:
                assert destination in components, f"Component '{destination}' is not in the project."
                destination_component = getattr(project.components, destination)
                assert nodes_can_match(origin_component, destination_component), \
                    f"""Components '{origin}' and '{destination}' cannot link their nodes:
                        \nNodes origin: {origin_component.nodes.value} 
                        \nNodes destination: {destination_component.nodes.value}"""

    # Assert all nodes of connected components match
    for n, matched in nodes_matched.items():
        assert matched, f"Node '{n[1]}' in component '{n[0]}' was not matched."