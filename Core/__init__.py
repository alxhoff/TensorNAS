from enum import Enum
import numpy as np

class EnumWithNone(str, Enum):
    def value(self):
        ret = self._value_
        if ret == "None":
            return None
        else:
            return ret

multiplier = {
    'GB': 1 / 1024**3,     # memory unit gega-byte
    'MB': 1 / 1024**2,     # memory unit mega-byte
    'MFLOPs': 1 / 10**8,   # FLOPs unit million-flops
    'BFLOPs': 1 / 10**11,  # FLOPs unit billion-flops
    'Million': 1 / 10**6,  # paprmeter count unit millions
    'Billion': 1 / 10**9,  # paprmeter count unit billions
}


def multiply(fn):
    def deco(units, *args, **kwds):
        return np.round(multiplier.get(units, -1) * fn(*args, **kwds), 4)
    return deco

def count_linear(layers):
    MAC = layers.output_shape[1] * layers.input_shape[1]
    try:
        if layers.get_config()["use_bias"]:
            ADD = layers.output_shape[1]
        else:
            ADD = 0
    except KeyError:
        ADD = 0
    return MAC*2 + ADD

def count_conv2d(layers, log = False):
    
    if layers.output_shape[1] != None:
        numshifts = int(layers.output_shape[1] * layers.output_shape[2])
    elif layers.output_shape[1] == None:
        numshifts = int(layers.output_shape[-1])
    
    # MAC/convfilter = kernelsize^2 * InputChannels * OutputChannels
    try:
        MACperConv = layers.get_config()["kernel_size"][0] * layers.get_config()["kernel_size"][1] * layers.input_shape[3] * layers.output_shape[3]
    except KeyError:
        MACperConv = 0
        pass
    
    try:
        if layers.get_config()["use_bias"]:
            ADD = layers.output_shape[3]
        else:
            ADD = 0
    except KeyError:
        ADD = 0
        pass
        
    return MACperConv * numshifts * 2 + ADD



@multiply
def count_flops(model, log = False):
    '''
    ParametersNo documen
    ----------
    model : A keras or TF model
    Returns
    -------
    Sum of all layers FLOPS in unit scale, you can convert it 
    afterward into Millio or Billio FLOPS
    '''
    layer_flops = []
    # run through models
    if isinstance(model, list):
        for layer in model:
            if "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"] or "squeeze" in layer.get_config()["name"]:
                layer_flops.append(count_linear(layer))
            elif "conv" in layer.get_config()["name"] :
                layer_flops.append(count_conv2d(layer,log))
            elif "dwconv" in layer.get_config()["name"]:
                layer_flops.append(count_conv2d(layer,log))
            elif "expand" in layer.get_config()["name"]:
                layer_flops.append(count_conv2d(layer,log))
            elif "res" in layer.get_config()["name"]:
                layer_flops.append(count_conv2d(layer,log))
            elif "stage" in layer.get_config()['name']:
                layer_flops.append(count_conv2d(layer,log))
    else:    
        for layer in model.layers:
            if "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"] or "squeeze" in layer.get_config()["name"]:
                layer_flops.append(count_linear(layer))
            elif "conv" in layer.get_config()["name"]:
                layer_flops.append(count_conv2d(layer,log))
            elif "dwconv" in layer.get_config()["name"]:
                layer_flops.append(count_conv2d(layer,log))
            elif "expand" in layer.get_config()["name"]:
                layer_flops.append(count_conv2d(layer,log))
            elif "res" in layer.get_config()["name"]:
                layer_flops.append(count_conv2d(layer,log))
            elif "stage" in layer.get_config()['name']:
                layer_flops.append(count_conv2d(layer,log))
    
    return np.sum(layer_flops, dtype=np.int64, initial=0)