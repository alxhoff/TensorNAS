from demos.mnistdemoinput import input_tensor_shape, mnist_class_count
from tensornas.blocktemplates.blockarchitectures import classificationblockarchitectures
from tensornas.core.crossover import _select_random_node

print("###########################")
print("First model architecture\n")

model1 = classificationblockarchitectures.ClassificationBlockArchitecture(
    input_tensor_shape, mnist_class_count
)
print(model1.get_ascii_tree() + "\n\n")

print("###########################")
print("Second model architecture\n")

model2 = classificationblockarchitectures.ClassificationBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

print(model2.get_ascii_tree() + "\n\n")

print("###########################")
print("First model random node\n")

rn1 = _select_random_node(model1)
print(
    "From index {} in parent {}\n".format(
        rn1.get_index_in_parent(), rn1.parent_block._get_name()
    )
)
print(rn1.get_ascii_tree() + "\n")

print("###########################")
print("Second model random node\n")

rn2 = _select_random_node(model2)
print(
    "From index {} in parent {}\n".format(
        rn2.get_index_in_parent(), rn2.parent_block._get_name()
    )
)
print(rn2.get_ascii_tree() + "\n")

print("###########################")
print("Swapping nodes in models to create crossed-over children")
ind1 = rn1.get_index_in_parent()
ind2 = rn2.get_index_in_parent()

parent1 = rn1.parent_block
parent2 = rn2.parent_block

rn1.parent_block = parent2
rn2.parent_block = parent1

parent1.set_block_at_index(ind1, rn2)
parent2.set_block_at_index(ind2, rn1)


print("###########################")
print("First model architecture\n")

print(model1.get_ascii_tree() + "\n\n")

print("###########################")
print("Second model architecture\n")
print(model2.get_ascii_tree() + "\n\n")
