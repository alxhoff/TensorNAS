# Note: please take note of arguments and return forms!
def crossover_individuals_sp(ind1, ind2):
    """
    A pythonic approach to crossing over, as invalid architectures are created by some crossovers, crossing over is
    repeated until a valid architecture is produced.
    """
    from copy import deepcopy
    from TensorNAS.Core.Crossover import crossover_single_point

    while True:
        try:
            ind3, ind4 = deepcopy(ind1), deepcopy(ind2)
            ind3.block_architecture, ind4.block_architecture = crossover_single_point(
                ind3.block_architecture, ind4.block_architecture
            )
        except Exception:
            continue
        break

    ind3.index = None
    ind4.index = None
    return ind3, ind4


def crossover_single_point(b1, b2):
    """
    A single block between the two architectures is swapped. Care should be taken as the crossover can raise exceptions
    if an invalid architecture is generated.

    @param b1 First block architecture to be crossed-over
    @param b2 Second block architecture to be crossed-over
    """
    random_node_1 = _select_random_node(b1)
    random_node_2 = _select_random_node(b2)

    index_1 = random_node_1.get_index_in_parent()
    index_2 = random_node_2.get_index_in_parent()

    parent_1 = random_node_1.parent_block
    parent_2 = random_node_2.parent_block

    random_node_1.parent_block = parent_2
    random_node_2.parent_block = parent_1

    parent_1.set_block_at_index(index_1, random_node_2)
    parent_2.set_block_at_index(index_2, random_node_1)

    random_node_1.reset_ba_input_shapes()
    random_node_2.reset_ba_input_shapes()

    return b1, b2


def _get_max_depth(ba):
    from TensorNAS.Core.Layer import Layer

    if isinstance(ba, Layer):
        return 0

    if ba.input_blocks or ba.middle_blocks or ba.output_blocks:

        for sb in ba.input_blocks + ba.middle_blocks + ba.output_blocks:
            return _get_max_depth(sb) + 1


def _get_middle_nodes(ba, depth):
    """
    Returns a list of the nodes at a certain hierarchical depth of a block architecture.

    @param ba Input block architecture
    @param depth Hierarchical depth to be retrieved
    @param block_list List that is to store the result, necessary for recursion

    @return List of blocks
    """
    depth -= 1

    if not depth:
        return ba.middle_blocks
    else:
        nodes = []
        for sb in ba.middle_blocks:
            nodes.extend(_get_middle_nodes(sb, depth))
    return nodes


def crossover_cutting_point(b1, b2, depth=1):
    """
    Find a random point in both block architectures and cut the block architecture, swapping the halves. This is
    done in respect to a specified depth, meaning that the block architectures are cut on a specified hierarchical
    depth.

    Block architectures are only ever cut in their middle blocks, such that input and output constraints are never
    violated.
    """
    from random import randint

    assert depth > 0

    depth_1 = _get_max_depth(b1)
    depth_2 = _get_max_depth(b2)

    if depth > depth_1 or depth > depth_2:
        return b1, b2

    tier_nodes_1 = _get_middle_nodes(b1, depth)
    tier_nodes_2 = _get_middle_nodes(b2, depth)

    cut_point_1 = randint(0, len(tier_nodes_1) - 1)
    cut_point_2 = randint(0, len(tier_nodes_2) - 1)

    # merge contents of parent nodes
    node_1 = tier_nodes_1[cut_point_1]
    node_2 = tier_nodes_2[cut_point_2]
    parent_1 = node_1.parent_block
    parent_2 = node_2.parent_block
    node_1_index = node_1.get_middle_index_in_parent()
    node_2_index = node_2.get_middle_index_in_parent()

    # Nodes from before the cut are swapped
    nodes_from_1_to_move = parent_1.middle_blocks[: node_1_index + 1]
    nodes_from_2_to_move = parent_2.middle_blocks[: node_2_index + 1]

    del parent_1.middle_blocks[: node_1_index + 1]
    del parent_2.middle_blocks[: node_2_index + 1]

    parent_1.middle_blocks = nodes_from_2_to_move + parent_1.middle_blocks
    for node in nodes_from_2_to_move:
        node.parent_block = parent_1

    parent_2.middle_blocks = nodes_from_1_to_move + parent_2.middle_blocks
    for node in nodes_from_1_to_move:
        node.parent_block = parent_2

    parent_1.reset_ba_input_shapes()
    parent_2.reset_ba_input_shapes()

    return b1, b2


def _cross_point(b1, b2, b1_i, b2_i):
    """
    Crosses the b1_i indexed block from b1 with the b2_i indexed block from b2.

    @param b1 First block architecture to be crossedover
    @param b2 Second block architecture to be crossedover
    @param b1_i Index of block from b1 to be crossedover
    @param b2_i Index of block from b2 to be crossedover

    @return Tuple containing b1 and b2
    """
    # TODO
    return b1, b2


def _recurse_select(block, count):
    import random
    from TensorNAS.Core.Block import Block

    ret = None
    if random.randint(0, count) == count:
        ret = block
    count += 1

    for sb in [
        b
        for b in block.input_blocks + block.middle_blocks + block.output_blocks
        if issubclass(type(b), Block)
    ]:
        tmp, count = _recurse_select(sb, count)
        if tmp:
            ret = tmp

    return ret, count


def _select_random_node(ba):
    """
    Algorithm: Start at first node in tree, iterate through tree, keeping a count of current steps taken. At each node
    pick a random number between [0, current count], eg. second node would be a exclusive range between 0 and 2. If the
    the generated value is `current count - 1` then set that node as the selected node. Continue until there are no
    more nodes. Due to the decreasing probability as the count increases, the overall probability for each node is
    equal.

    @param ba The block architecture
    """
    count = 0
    while True:
        selected_block, count = _recurse_select(ba, count)
        if selected_block is None or selected_block == ba:
            count = 0
        else:
            return selected_block
