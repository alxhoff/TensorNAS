from TensorNAS.Core.Mutate import (
    mutate_int,
    mutate_tuple,
    mutate_enum,
    MutationOperators,
)


def layer_mutation(func):
    def wrapper(self, **kwargs):
        mutation_table_ref = [self.mutation_table.get_mutation_table_ref(func.__name__)]
        return func.__name__, func(self, **kwargs), mutation_table_ref

    return wrapper


class MutateFilters:
    @layer_mutation
    def _mutate_filters_up(self, operator=MutationOperators.STEP_UP):

        prev_filters = self.args[self.get_args_enum().FILTERS]
        self.args[self.get_args_enum().FILTERS] = mutate_int(
            self.args[self.get_args_enum().FILTERS],
            1,
            self.MAX_FILTER_COUNT,
            operator=operator,
        )
        return "Mutated filters {} -> {}".format(
            prev_filters, self.args[self.get_args_enum().FILTERS]
        )

    @layer_mutation
    def _mutate_filters_down(self, operator=MutationOperators.STEP_DOWN):
        prev_filters = self.args[self.get_args_enum().FILTERS]
        self.args[self.get_args_enum().FILTERS] = mutate_int(
            self.args[self.get_args_enum().FILTERS],
            1,
            self.MAX_FILTER_COUNT,
            operator=operator,
        )
        return "Mutated filters {} -> {}".format(
            prev_filters, self.args[self.get_args_enum().FILTERS]
        )


class MutateKernelSize:
    @layer_mutation
    def _mutate_kernel_size_up(self, operator=MutationOperators.SYNC_STEP_UP):

        prev_kernel_size = self.args[self.get_args_enum().KERNEL_SIZE]
        self.args[self.get_args_enum().KERNEL_SIZE] = mutate_tuple(
            self.args[self.get_args_enum().KERNEL_SIZE],
            1,
            self.MAX_KERNEL_DIMENSION,
            operator=operator,
        )
        return "Mutated kernel size {} -> {}".format(
            prev_kernel_size, self.args[self.get_args_enum().KERNEL_SIZE]
        )

    @layer_mutation
    def _mutate_kernel_size_down(self, operator=MutationOperators.SYNC_STEP_DOWN):
        prev_kernel_size = self.args[self.get_args_enum().KERNEL_SIZE]
        self.args[self.get_args_enum().KERNEL_SIZE] = mutate_tuple(
            self.args[self.get_args_enum().KERNEL_SIZE],
            1,
            self.MAX_KERNEL_DIMENSION,
            operator=operator,
        )
        return "Mutated kernel size {} -> {}".format(
            prev_kernel_size, self.args[self.get_args_enum().KERNEL_SIZE]
        )


class MutateStrides:
    @layer_mutation
    def _mutate_strides_up(self, operator=MutationOperators.SYNC_STEP_UP):

        prev_strides = self.args[self.get_args_enum().STRIDES]
        self.args[self.get_args_enum().STRIDES] = mutate_tuple(
            self.args[self.get_args_enum().STRIDES],
            1,
            self.MAX_STRIDE,
            operator=operator,
        )
        return "Mutating strides {} -> {}".format(
            prev_strides, self.args[self.get_args_enum().STRIDES]
        )

    @layer_mutation
    def _mutate_strides_down(self, operator=MutationOperators.SYNC_STEP_DOWN):

        prev_strides = self.args[self.get_args_enum().STRIDES]
        self.args[self.get_args_enum().STRIDES] = mutate_tuple(
            self.args[self.get_args_enum().STRIDES],
            1,
            self.MAX_STRIDE,
            operator=operator,
        )
        return "Mutating strides {} -> {}".format(
            prev_strides, self.args[self.get_args_enum().STRIDES]
        )


class MutatePadding:
    @layer_mutation
    def _mutate_padding_up(self, operator=MutationOperators.SYNC_STEP_UP):
        from TensorNAS.Core.Layer import ArgPadding

        prev_padding = self.args[self.get_args_enum().PADDING]
        self.args[self.get_args_enum().PADDING] = mutate_enum(
            self.args[self.get_args_enum().PADDING], ArgPadding
        )
        return "Mutating padding {} -> {}".format(
            prev_padding, self.args[self.get_args_enum().PADDING]
        )

    @layer_mutation
    def _mutate_padding_down(self, operator=MutationOperators.SYNC_STEP_DOWN):
        from TensorNAS.Core.Layer import ArgPadding

        prev_padding = self.args[self.get_args_enum().PADDING]
        self.args[self.get_args_enum().PADDING] = mutate_enum(
            self.args[self.get_args_enum().PADDING], ArgPadding
        )
        return "Mutating padding {} -> {}".format(
            prev_padding, self.args[self.get_args_enum().PADDING]
        )


class MutateDilationRate:
    @layer_mutation
    def _mutate_dilation_rate_up(self, operator=MutationOperators.SYNC_STEP_UP):

        prev_dilation_rate = self.args[self.get_args_enum().DILATION_RATE]
        self.args[self.get_args_enum().DILATION_RATE] = mutate_tuple(
            self.args[self.get_args_enum().DILATION_RATE],
            1,
            self.MAX_DILATION,
            operator=operator,
        )
        return "Mutating dilation rate {} -> {}".format(
            prev_dilation_rate, self.args[self.get_args_enum().DILATION_RATE]
        )

    @layer_mutation
    def _mutate_dilation_rate_down(self, operator=MutationOperators.SYNC_STEP_DOWN):

        prev_dilation_rate = self.args[self.get_args_enum().DILATION_RATE]
        self.args[self.get_args_enum().DILATION_RATE] = mutate_tuple(
            self.args[self.get_args_enum().DILATION_RATE],
            1,
            self.MAX_DILATION,
            operator=operator,
        )
        return "Mutating dilation rate {} -> {}".format(
            prev_dilation_rate, self.args[self.get_args_enum().DILATION_RATE]
        )


class MutateActivation:
    @layer_mutation
    def _mutate_activation(self):
        from TensorNAS.Core.Layer import ArgActivations

        prev_activation = self.args[self.get_args_enum().ACTIVATION]
        self.args[self.get_args_enum().ACTIVATION] = mutate_enum(
            self.args[self.get_args_enum().ACTIVATION],
            ArgActivations,
        )
        return "Mutating activation {} -> {}".format(
            prev_activation, self.args[self.get_args_enum().ACTIVATION]
        )


class MutatePoolSize:
    @layer_mutation
    def _mutate_pool_size_up(self, operator=MutationOperators.SYNC_STEP_UP):

        prev_pool_size = self.args[self.get_args_enum().POOL_SIZE]
        self.args[self.get_args_enum().POOL_SIZE] = mutate_tuple(
            self.args[self.get_args_enum().POOL_SIZE],
            1,
            self.MAX_POOL_SIZE,
            operator=operator,
        )
        return "Mutating pool size {} -> {}".format(
            prev_pool_size, self.args[self.get_args_enum().POOL_SIZE]
        )

    @layer_mutation
    def _mutate_pool_size_down(self, operator=MutationOperators.SYNC_STEP_DOWN):

        prev_pool_size = self.args[self.get_args_enum().POOL_SIZE]
        self.args[self.get_args_enum().POOL_SIZE] = mutate_tuple(
            self.args[self.get_args_enum().POOL_SIZE],
            1,
            self.MAX_POOL_SIZE,
            operator=operator,
        )
        return "Mutating pool size {} -> {}".format(
            prev_pool_size, self.args[self.get_args_enum().POOL_SIZE]
        )


class MutateUnits:
    @layer_mutation
    def _mutate_units_up(self, operator=MutationOperators.STEP_UP):
        from TensorNAS.Core.Mutate import mutate_int

        prev_units = self.args[self.get_args_enum().UNITS]
        self.args[self.get_args_enum().UNITS] = mutate_int(
            self.args.get(self.get_args_enum().UNITS),
            1,
            self.MAX_UNITS,
            operator=operator,
        )
        return "Mutating units {} -> {}".format(
            prev_units, self.args[self.get_args_enum().UNITS]
        )

    @layer_mutation
    def _mutate_units_down(self, operator=MutationOperators.STEP_DOWN):
        from TensorNAS.Core.Mutate import mutate_int

        prev_units = self.args[self.get_args_enum().UNITS]
        self.args[self.get_args_enum().UNITS] = mutate_int(
            self.args.get(self.get_args_enum().UNITS),
            1,
            self.MAX_UNITS,
            operator=operator,
        )
        return "Mutating units {} -> {}".format(
            prev_units, self.args[self.get_args_enum().UNITS]
        )


class MutateNumGroups:
    @layer_mutation
    def _mutate_num_groups_up(self, operator=MutationOperators.STEP_UP):
        from TensorNAS.Core.Mutate import mutate_int

        prev_num_groups = self.args[self.get_args_enum().NUM_GROUPS]
        self.args[self.get_args_enum().NUM_GROUPS] = mutate_int(
            self.args.get(self.get_args_enum().NUM_GROUPS),
            1,
            self.MAX_NUM_GROUPS,
            operator=operator,
        )
        return "Mutating num groups {} -> {}".format(
            prev_num_groups, self.args[self.get_args_enum().NUM_GROUPS]
        )

    @layer_mutation
    def _mutate_num_groups_down(self, operator=MutationOperators.STEP_DOWN):
        from TensorNAS.Core.Mutate import mutate_int

        prev_num_groups = self.args[self.get_args_enum().NUM_GROUPS]
        self.args[self.get_args_enum().NUM_GROUPS] = mutate_int(
            self.args.get(self.get_args_enum().NUM_GROUPS),
            1,
            self.MAX_NUM_GROUPS,
            operator=operator,
        )
        return "Mutating num groups {} -> {}".format(
            prev_num_groups, self.args[self.get_args_enum().NUM_GROUPS]
        )


class MutateTargetShape:
    @layer_mutation
    def _mutate_target_shape(self):
        from TensorNAS.Core.Mutate import mutate_dimension

        prev_target_shape = self.args[self.get_args_enum().TARGET_SHAPE]
        self.args[self.get_args_enum().TARGET_SHAPE] = mutate_dimension(
            self.args[self.get_args_enum().TARGET_SHAPE]
        )
        return "Mutating target shape {} -> {}".format(
            prev_target_shape, self.args[self.get_args_enum().TARGET_SHAPE]
        )


class MutateRate:
    @layer_mutation
    def _mutate_rate(self):
        from TensorNAS.Core.Mutate import mutate_float

        prev_rate = self.args[self.get_args_enum().RATE]
        self.args[self.get_args_enum().RATE] = mutate_float(
            self.args[self.get_args_enum().RATE], 0, self.MAX_RATE
        )
        return "Mutating rate {} -> {}".format(
            prev_rate, self.args[self.get_args_enum().RATE]
        )


def _single_stride(self):
    st = self.args[self.get_args_enum().STRIDES]
    if st[0] == 1 and st[1] == 1:
        return True
    return False


def _single_dilation_rate(self):
    dr = self.args[self.get_args_enum().DILATION_RATE]
    if dr[0] == 1 and dr[1] == 1:
        return True
    return False
