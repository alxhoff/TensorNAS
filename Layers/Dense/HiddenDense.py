from TensorNAS.Layers.Dense import Layer
from TensorNAS.Core.LayerMutations import MutateUnits, MutateActivation


class Layer(Layer, MutateUnits, MutateActivation):
    MAX_UNITS = 256
