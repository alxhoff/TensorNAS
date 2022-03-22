from TensorNAS.Core.Mutate import (
    mutate_int,
    mutate_tuple,
    mutate_enum,
    MutationOperators,
)


class MutateFilters:
    def _mutate_filters(self, operator=MutationOperators.STEP):

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
    def _mutate_kernel_size(self, operator=MutationOperators.SYNC_STEP):

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
    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):

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
    def _mutate_padding(self):
        from TensorNAS.Core.Layer import ArgPadding

        prev_padding = self.args[self.get_args_enum().PADDING]
        self.args[self.get_args_enum().PADDING] = mutate_enum(
            self.args[self.get_args_enum().PADDING], ArgPadding
        )
        return "Mutating padding {} -> {}".format(
            prev_padding, self.args[self.get_args_enum().PADDING]
        )


class MutateDilationRate:
    def _mutate_dilation_rate(self, operator=MutationOperators.SYNC_STEP):

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
    def _mutate_pool_size(self, operator=MutationOperators.SYNC_STEP):

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
    def _mutate_units(self):
        from TensorNAS.Core.Mutate import mutate_int

        prev_units = self.args[self.get_args_enum().UNITS]
        self.args[self.get_args_enum().UNITS] = mutate_int(
            self.args.get(self.get_args_enum().UNITS), 1, self.MAX_UNITS
        )
        return "Mutating units {} -> {}".format(
            prev_units, self.args[self.get_args_enum().UNITS]
        )


class MutateNumGroups:
    def _mutate_num_groups(self):
        from TensorNAS.Core.Mutate import mutate_int

        prev_num_groups = self.args[self.get_args_enum().NUM_GROUPS]
        self.args[self.get_args_enum().NUM_GROUPS] = mutate_int(1, self.MAX_NUM_GROUPS)
        return "Mutating num groups {} -> {}".format(
            prev_num_groups, self.args[self.get_args_enum().NUM_GROUPS]
        )


class MutateTargetShape:
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
    def _mutate_rate(self):
        from TensorNAS.Core.Mutate import mutate_unit_interval

        prev_rate = self.args[self.get_args_enum().RATE]
        self.args[self.get_args_enum().RATE] = mutate_unit_interval(
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
