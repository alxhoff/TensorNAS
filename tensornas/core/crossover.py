def crossover_blocks(b1, b2):
    pass


def crossover_single_point(b1, b2):
    """
    A single block between the two architectures is swapped.
    """
    random_node_1 = _select_random_node(b1)
    random_node_2 = _select_random_node(b2)

    index_1 = random_node_1.get_index_in_parent()
    index_2 = random_node_2.get_index_in_parent()

    parent_1 = random_node_1.parent_block
    parent_2 = random_node_2.parent_block

    parent_1.set_block_at_index(index_1, random_node_2)
    parent_2.set_block_at_index(index_2, random_node_1)

    return b1, b2


def crossover_cutting_point(b1, b2):
    """
    Find a random point in both block architectures.
    """
    pass


def _cross_point(b1, b2, b1_i, b2_i):
    """
    Crosses the b1_i indexed block from b1 with the b2_i indexed block from b2.

    @param b1 First block architecture to be crossedover
    @param b2 Second block architecture to be crossedover
    @param b1_i Index of block from b1 to be crossedover
    @param b2_i Index of block from b2 to be crossedover

    @return Tuple containing b1 and b2
    """
    return (b1, b2)


def _recurse_select(block, count):
    import random

    ret = None
    if random.randint(0, count) == count:
        ret = block
    count += 1
    for sb in block.input_blocks + block.middle_blocks + block.output_blocks:
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
