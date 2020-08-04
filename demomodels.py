from tensornasjson import *
from tensorflowlayerargs import *

Model_Configuration1 = {
    0: {
        "name": "Conv2D",
        "args": {
            Conv2DArgs.FILTERS.name: 28,
            Conv2DArgs.KERNEL_SIZE.name: [3, 3],
            Conv2DArgs.INPUT_SIZE.name: [28, 28, 1],
        },
    },
    1: {"name": "MaxPool2D", "args": {MaxPool2DArgs.POOL_SIZE.name: [2, 2],}},
    2: {"name": "Flatten"},
    3: {
        "name": "Dense",
        "args": {DenseArgs.UNITS.name: 128, DenseArgs.ACTIVATION.name: "relu"},
    },
    4: {"name": "Dropout", "args": {DropoutArgs.RATE.name: 0.2}},
    5: {
        "name": "Dense",
        "args": {DenseArgs.UNITS.name: 10, DenseArgs.ACTIVATION.name: "softmax"},
    },
}

Model_Configuration2 = {
    0: {
        "name": "Conv2D",
        "args": {
            Conv2DArgs.FILTERS.name: 12,
            Conv2DArgs.KERNEL_SIZE.name: [3, 3],
            Conv2DArgs.STRIDES.name: [2, 2],
            Conv2DArgs.INPUT_SIZE.name: [28, 28, 1],
        },
    },
    1: {"name": "Reshape", "args": {ReshapeArgs.TARGET_SHAPE.name: [13, 13, 12, 1]}},
    2: {"name": "MaxPool3D", "args": {MaxPool2DArgs.POOL_SIZE.name: [6, 3, 3],}},
    3: {"name": "Flatten"},
    4: {
        "name": "Dense",
        "args": {DenseArgs.UNITS.name: 10, DenseArgs.ACTIVATION.name: "softmax"},
    },
}


def generate_demo_model_jsons():
    write_nas_model_json(Model_Configuration1, "demo_model_1")
    write_nas_model_json(Model_Configuration2, "demo_model_2")


def generate_demo_model_array():
    return [load_nas_model_json("demo_model_1"), load_nas_model_json("demo_model_2")]
