### FROM JACK HAWTHORNE MASTER THESIS

from . import sets
from . import params
from . import constraints
from . import variables
from . import objectives


def set_up(network, objs):
    # create the model instances
    pyomo_tools = network.project.pyomo_tools
    ConcreteModel = pyomo_tools['ConcreteModel']
    model = ConcreteModel()
    # apply sets, params, varaibles, constraints and objectives to each model
    model = sets.set_up(network, model)
    model = params.set_up(network, model, objs)
    model = variables.set_up(network, model, objs)
    model = constraints.set_up(network, model)
    model = objectives.set_up(network, model, objs)
    return model


if __name__ == "__main__":
    pass
