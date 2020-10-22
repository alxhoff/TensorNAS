from demos.mnistdemoinput import input_tensor_shape, mnist_class_count
from tensornas.blocktemplates.blockarchitectures import classificationblockarchitectures
from tensornas.core.crossover import _select_random_node

model1 = classificationblockarchitectures.ClassificationBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

rn = _select_random_node(model1)

rn.print()
