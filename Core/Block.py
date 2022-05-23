import random
import re
from abc import ABC, abstractmethod
from enum import Enum, auto

from TensorNAS.Core.Mutate import mutate_enum_i


def get_block_from_JSON(json_dict, parent_block=None):
    class_name = json_dict["class_name"]

    from TensorNAS.Tools.JSONImportExport import GetBlockMod

    b_class = GetBlockMod(class_name).Block

    import inspect

    class_args = inspect.getfullargspec(b_class.__init__).args

    class_args = [
        json_dict[key] if key != "parent_block" else parent_block
        for key in class_args[1:]
    ]

    blk = b_class(*class_args)
    blk.input_blocks = []
    blk.middle_blocks = []
    blk.output_blocks = []
    blk = _import_subblocks_from_json(blk, json_dict)

    return blk


def _import_subblocks_from_json(blk, json_dict):

    for i, b in enumerate(json_dict["input_blocks"]):
        blk.input_blocks.append(get_block_from_JSON(b, blk))

    for i, b in enumerate(json_dict["middle_blocks"]):
        blk.middle_blocks.append(get_block_from_JSON(b, blk))

    for i, b in enumerate(json_dict["output_blocks"]):
        blk.output_blocks.append(get_block_from_JSON(b, blk))

    return blk


class MutationTable:
    def __init__(self, cls_obj):

        # Class name for debugging purposes
        self.class_name = str(cls_obj.__class__)  # .__bases__
        self.mutations = {}

        if hasattr(cls_obj, "mutation_funcs"):
            self.mutation_funs = cls_obj.mutation_funcs

    def get_mutation_table_ref(self, mutation):

        if not mutation in self.mutations:
            self.mutations[mutation] = [0, 0]

        ret = self.mutations[mutation]

        return ret

    def get_mutation_probability(self, function_name, index=0):
        """
        Choosing the Q value is then done by taking the tanh of each Q, normalizing and shifting to fit between
        0 and 1, thus, P = 0.5 * tanh(Q) + 0.5
        """
        from numpy import tanh as th

        q = self.mutations.get(function_name, (0, 0))[index]

        ret = 0.5 * th(q) + 0.5

        return ret


def add_mutation_table(func):
    """
    Function decorator used to decorate the block init function to create a PER-CLASS mutation table
    """

    def wrapper(self, input_shape, parent_block, args=None):
        func(self, input_shape, parent_block, args)
        if not hasattr(self.__class__, "mutation_table"):
            setattr(self.__class__, "mutation_table", MutationTable(self))

    return wrapper


class BaseBlock(ABC):
    """
    An abstract class that all model blocks are derived from. Thus all model blocks, regardless of their depth within
    the model architecture binary tree they must implement all of the abstract methods defined within this class.

    Required properties:
        - SUB_BLOCK_TYPES

    Required (abstract) methods:
        - Generate random sub block

    Optional methods:
        - Generate constrained output sub blocks
        - Generate constrained input sub blocks
        - Mutate self
        - Get keras model
        - Print self
        - Output shape
        - Check new layer type
        - Repair self

    Should not be overridden:
        - Repair
        - Mutate
        - __init__
    """

    """
    A property to specify a minimum block count, is not required by each sub-class.
    """
    MIN_SUB_BLOCKS = 0
    MAX_SUB_BLOCKS = 0

    @property
    @classmethod
    @abstractmethod
    class SubBlocks(Enum):
        """The enum type storing all the possible middle sub-blocks is required to be passed to the child class as it is
        used during random selection of sub-block blocks
        """

        NONE = auto()

    @add_mutation_table
    def __init__(self, input_shape, parent_block, args=None):
        """
        The init sequence of the Block class should always be called at the end of a subclass's __init__, via
        super().__init__ if a subclass is to implement its own __init__ method.

        This can be required if the block needs to take in additional parameters when being created, eg. a classification
        block needs to know the number of output classes that it must classify. Thus, such an implementation will
        read in its required init arguments and then invoke the Block.__init__ such that the defined Block init
        sequence is performed.
        """
        self.input_shape = input_shape
        self.parent_block = parent_block
        self.args_enum = self._get_args_enum()
        self.args = args

        if args:
            if isinstance(args, list):
                args = dict(args)
                new_dict = {}

                for key, val in args.items():
                    if isinstance(key, str):
                        new_dict[
                            [i for i in self.args_enum if i.name == key][0]
                        ] = args[key]
                args = new_dict

        try:
            self.mutation_funcs = [
                func
                for func in dir(self)
                if callable(getattr(self, func))
                and re.search(r"^_mutate(?!_self)", func)
            ]

        except Exception as e:
            raise e

        self.input_blocks = []
        self.middle_blocks = []
        self.output_blocks = []

        if args:
            ib = self.generate_constrained_input_sub_blocks(
                input_shape=input_shape, args=args
            )
        else:
            ib = self.generate_constrained_input_sub_blocks(input_shape=input_shape)

        if ib:
            self.input_blocks.extend(ib)

        if args:
            mb = self.generate_constrained_middle_sub_blocks(
                input_shape=self._get_cur_output_shape(), args=args
            )
        else:
            mb = self.generate_constrained_middle_sub_blocks(
                input_shape=self._get_cur_output_shape()
            )

        if mb:
            self.middle_blocks.extend(mb)

        if self.MAX_SUB_BLOCKS:
            self._generate_sub_blocks()

        if args:
            ob = self.generate_constrained_output_sub_blocks(
                input_shape=self._get_cur_output_shape(), args=args
            )
        else:
            ob = self.generate_constrained_output_sub_blocks(
                input_shape=self._get_cur_output_shape()
            )

        if ob:
            self.output_blocks.extend(ob)

    def _invoke_random_mutation_function(self, mutation_goal_index=0, verbose=False):
        if self.mutation_funcs:
            weights = [
                self.mutation_table.get_mutation_probability(func)
                for func in self.mutation_funcs
            ]
            try:
                func_name = random.choices(self.mutation_funcs, weights=weights)[0]
            except Exception as e:
                print(e)
            mutate_eval = "self." + func_name
            if verbose == True:
                print("[MUTATE] invoking `{}`".format(mutate_eval))
            return eval(mutate_eval)()
        return "Null", None

    def mutate_self(self, mutation_goal_index=0, verbose=False):
        """
        An optional function that allows for the block to mutate itself during mutation, by default this function
        simply invokes mutation of a random mutation function and if that is not possible then
         random mutation of a sub block by invoking mutate_subblock
        """
        if len(self.mutation_funcs) > 0:
            return self._invoke_random_mutation_function(
                mutation_goal_index=mutation_goal_index, verbose=verbose
            )

    def mutate(
        self,
        mutation_goal_index=0,
        mutate_equally=True,
        mutation_probability=0.0,
        verbose=False,
    ):
        """Similar to NetworkLayer objects, block mutation is a randomized call to any methods prexied with `_mutate`,
        this includes the defaul `mutate_subblock`.

        The implementation of a block should as such then present the possible mutation possibilities as a collection
        of `_mutate` functions. Generally mutation will call the default `mutate_subblock` method to invoke mutation
        in a randomly selected sub-block.

        If specific mutation operations are thus required they can be implemented. Among the default mutation functions
        is the `mutate_self` function which directly mutates the block instead of calling mutate in a sub-block.
        The function by default does nothing and returns False, in such a case another mutate function is called.
        If one wishes to implement `mutate_self` then it should return True to stop the subsequent
        re-invoking of mutate.

        The probability of mutating the block itself instead of it's sub-block is passed in via mutation_probability."""

        if mutate_equally:
            block = self._get_random_sub_block_inc_self()
            ret = block.mutate_self(
                mutation_goal_index=mutation_goal_index, verbose=verbose
            )
        else:
            prob = random.random()
            if (prob < mutation_probability) and (len(self.middle_blocks) > 0):
                # Mutate subblock
                if len(self.middle_blocks):
                    choice_index = random.choice(range(len(self.middle_blocks)))
                    if verbose:
                        print(
                            "[MUTATE] middle block #{} of type {}".format(
                                choice_index, type(self.middle_blocks[choice_index])
                            )
                        )
                    ret = self.middle_blocks[choice_index].mutate(
                        mutation_goal_index=mutation_goal_index,
                        mutate_equally=mutate_equally,
                        mutation_probability=mutation_probability,
                        verbose=verbose,
                    )
                # return format of all mutate functions, except most bottom level mutations, should be
                # function name, list of mutation table references, mutation note from bottom most level
                ret = tuple(["_mutate_subblock"] + list(ret[1:]))
            else:
                ret = self.mutate_self(
                    mutation_goal_index=mutation_goal_index, verbose=verbose
                )

        self.reset_ba_input_shapes()

        # Add the invoked mutation function to the mutation table by getting a reference to the operation in the
        # mutation table such that it can be populated with accuracy and param count once the mutation model is
        # evaluated
        mutation_function = ret[0]
        mutation_note = ret[1]

        # If this block is at the bottom of the block architecture hierarchy we need to create the list of mutation
        # table references to return
        mutation_table_ref = self.mutation_table.get_mutation_table_ref(
            mutation_function
        )
        if len(ret) < 3:
            table_ref_list = [mutation_table_ref]

        else:
            table_ref_list = ret[2]
            try:
                table_ref_list.append(mutation_table_ref)
            except Exception as e:
                print(e)

        # Return format is 'list of mutation table references', 'mutation note for logging'
        return mutation_function, mutation_note, table_ref_list

    def generate_constrained_output_sub_blocks(self, input_shape, args=None):
        """This method is called after the sub-blocks have been generated to generate the required blocks which are
        appended to the output_blocks list. An example of this would be the placement
        of a classification layer at the end of a model.

        @return Must return a list of created sub block objects
        """
        return []

    def generate_constrained_input_sub_blocks(self, input_shape, args=None):
        """This method is called before the sub-blocks have been generated to generate the required blocks which are
        appended to the input_blocks list. An example of this would be the placement
        of a convolution layer at the beginning of a model.

        @return Must return a list of created sub block objects
        """
        return []

    def generate_constrained_middle_sub_blocks(self, input_shape, args=None):
        """
        Different to constrained input and output sub blocks, mid sub blocks are up for mutation and can be modified.
        This function helps to generate a specific set of blocks originally instead of calling generate_random_sub_block
        which returns a random block.
        """
        return []

    def generate_random_sub_block(self, input_shape, subblock_type, args=None):
        """This method appends a randomly selected possible sub-block to the classes middle_blocks list, The block type is
        passed in as layer_type which is randomly selected from the provided enum SUB_BLOCK_TYPES which stores the
        possible sub block types. This function is responsible for instantiating each of these sub blocks if required.
        """
        return []

    def check_next_layer_type(self, prev_layer_type, next_layer_type):
        """
        This function is called when the next layer type is randomly generated, inside the function the user can perform
        checks to check if the next layer type is allowed, more likely will be checks to see if the next layer type is
        invalid, eg. two sequential flatten Layers.

        It should be noted that the previous layer is given as a SupportedLayer, this is an enum value generated by the
        TensorNAS Framework.Layers package by scanning the provided modules in the package. The next layer's type is given in
        terms of the possible sub-block types for the current block. As such you must make an apples to oranges
        comparison.
        """
        return True

    def _check_layer_types(self, next_layer_type):
        """
        An intermediate function that checks if there is a previous layer to be passed to the check_next_layer_type
        function.
        """
        if len(self.middle_blocks):
            return self.check_next_layer_type(
                type(self.middle_blocks[-1]), next_layer_type
            )
        return True

    def refresh_io_shapes(self, input_shape=None):
        """
        Recursive function that walks a block architecture, refreshing the input and output shapes of the architecture.

        @return True is no blocks were invalid
        """
        sbs = self.input_blocks + self.middle_blocks + self.output_blocks
        if len(sbs):
            if not input_shape:
                input_shape = self.get_input_shape()
            out_shape = input_shape
            for sb in sbs:
                sb.set_input_shape(out_shape)
                if sb.get_sb_count():
                    out_shape = sb.refresh_io_shapes(sb.input_shape)
                else:
                    out_shape = sb.get_output_shape()
                sb.set_output_shape(out_shape)
            self.set_output_shape(self.get_output_shape())
            return out_shape
        return self.get_output_shape()

    def reset_ba_input_shapes(self):
        """
        The block architecture root is retrieved and the sub-block inputs and outputs are processed and repaired.

        @return True if the change was successful, ie. no blocks became invalid
        """
        ba = self.get_block_architecture()
        ba.refresh_io_shapes(input_shape=ba.get_input_shape())
        return False

    def get_block_architecture(self):
        block = self
        while block.parent_block:
            block = block.parent_block

        return block

    def _generate_sub_blocks(self):
        """Subclasses of Block should not populate their sub-block lists but instead implement this function which
        will handle this. Generated blocks that are not valid
        """
        mb_count = len(self.middle_blocks)
        if self.MAX_SUB_BLOCKS and (self.MAX_SUB_BLOCKS > mb_count):
            rng = random.choice(
                range(
                    self.MIN_SUB_BLOCKS - mb_count, self.MAX_SUB_BLOCKS - mb_count + 1
                )
            )
            for i in range(rng):
                out_shape = self._get_cur_output_shape()
                blocks = self.generate_random_sub_block(
                    out_shape,
                    self._get_random_sub_block_type(),
                )
                if blocks:
                    self.middle_blocks.extend(blocks)

    def get_output_shape(self):
        """
        Returns the output shape of the block
        """
        if len(self.input_blocks + self.middle_blocks + self.output_blocks) >= 1:
            return (self.input_blocks + self.middle_blocks + self.output_blocks)[
                -1
            ].get_output_shape()
        else:
            return self.input_shape

    def set_output_shape(self, output_shape):
        self.output_shape = output_shape

    def get_input_shape(self):
        """
        Returns in the input shape of the block
        """
        return self.input_shape

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def _get_random_sub_block_type(self):
        """This method returns a random enum value of the block's possible sub-blocks"""
        if self.SubBlocks:
            while True:
                next_type = mutate_enum_i(self.SubBlocks)
                if self._check_layer_types(next_type):
                    return next_type
        else:
            return None

    def get_keras_layers(self, input_tensor):
        """By default this method simply calls this method in all child blocks, it should be overridden for layer
        blocks, ie. blocks that are leaves within the block hierarchy and contain a keras layer, such blocks should
        return an appropriately instantiated keras layer object

        TensorNAS Framework uses the Tensorflow functional API so each keras layer requires an input tensor, this shuold be
        passed from the previous layer.
        """
        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)

        return tmp

    def _get_cur_output_shape(self):
        if len(self.output_blocks):
            ret = self.output_blocks[-1].get_output_shape()
        elif len(self.middle_blocks):
            ret = self.middle_blocks[-1].get_output_shape()
        elif len(self.input_blocks):
            ret = self.input_blocks[-1].get_output_shape()
        else:
            ret = self.get_input_shape()
        try:
            assert ret, "Output shape is None"
            return ret
        except Exception as e:
            exit(e)

    def __str__(self):
        ret = ""
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            ret += str(sb)
        return ret

    def print(self):
        """
        Function useful for print debugging, the function by default invokes print self and then invokes printing in all
        child nodes. If you wish to print all children nodes then only override print_self and not print_self
        """
        self.print_self()
        print(str(self))

    def print_self(self):
        pass

    def _get_name(self):
        if hasattr(self, "layer"):
            return self.layer.__module__.split(".")[-1]
        return self.__module__.split(".")[-1]

    def get_ascii_tree(self):
        """
        Returns an ASCII tree representation of the block heirarchy starting from the current block.
        """
        from TensorNAS.Tools import stack_str_blocks
        from TensorNAS.Tools import block_width

        if not self.parent_block:
            name = "ROOT"
        else:
            name = self._get_name()

        io_str = " {}->{}".format(self.get_input_shape(), self.get_output_shape())
        name = "{" + name + io_str + "}"

        if not len(self.input_blocks + self.middle_blocks + self.output_blocks):
            return name

        child_strs = (
            [" |"]
            + [
                child.get_ascii_tree()
                for child in self.input_blocks + self.middle_blocks + self.output_blocks
            ]
            + ["| "]
        )
        child_widths = [block_width(s) for s in child_strs]

        display_width = max(
            len(name),
            sum(child_widths) + len(child_widths) - 1,
        )

        child_midpoints = []
        child_end = 0
        for width in child_widths:
            child_midpoints.append(child_end + (width // 2))
            child_end += width + 1

        brace_builder = []
        for i in range(display_width):
            if i < child_midpoints[0] or i > child_midpoints[-1]:
                brace_builder.append(" ")
            elif i in child_midpoints:
                brace_builder.append("+")
            else:
                brace_builder.append("-")
        brace = "".join(brace_builder)

        name_str = "{:^{}}".format(name, display_width)
        below = stack_str_blocks(child_strs)

        return name_str + "\n" + brace + "\n" + below

    def get_index_in_parent(self):
        if self.parent_block:
            return self.parent_block.get_block_index(self)
        return None

    def get_middle_index_in_parent(self):
        if self.parent_block:
            return self.parent_block.get_block_index_middle(self)
        return None

    def get_block_at_index(self, index):
        if len(self.input_blocks + self.middle_blocks + self.output_blocks) > (
            index + 1
        ):
            return None
        return (self.input_blocks + self.middle_blocks + self.output_blocks)[index]

    def get_block_index(self, block):
        for index, sb in enumerate(
            self.input_blocks + self.middle_blocks + self.output_blocks
        ):
            if block == sb:
                return index
        return None

    def _get_random_sub_block_inc_self(self):
        blocks = self._get_all_sub_blocks_inc_self()

        return random.choice(blocks)

    def _get_all_sub_blocks_inc_self(self):
        blocks = [self]

        for block in self.input_blocks + self.middle_blocks + self.output_blocks:
            from TensorNAS.Core.Layer import Layer

            if isinstance(block, Layer):
                return blocks
            else:
                blocks += block._get_all_sub_blocks_inc_self()

        return blocks

    def get_block_architecture(self):
        parent = self
        try:
            while parent.parent_block != None:
                parent = parent.parent_block
        except Exception as e:
            raise e

        return parent

    def get_block_index_middle(self, block):
        for index, sb in enumerate(self.middle_blocks):
            if block == sb:
                return index
        return None

    def set_block_at_index(self, index, block):
        if self.input_blocks:
            if index < len(self.input_blocks):
                self.input_blocks[index] = block
                return
            else:
                index -= len(self.input_blocks)

        if self.middle_blocks:
            if index < len(self.middle_blocks):
                self.middle_blocks[index] = block
                return
            else:
                index -= len(self.middle_blocks)

        if self.output_blocks:
            if index < len(self.output_blocks):
                self.output_blocks[index] = block

    def _get_random_subblock_class(self):
        return random.choice([e for e in self.SubBlocks if isinstance(e.value, int)])

    def get_sb_count(self):
        return len(self.input_blocks + self.middle_blocks + self.output_blocks)

    def _args_to_JSON(self):

        if self.args:
            args = dict(self.args)

            ret = []

            for arg in args:
                ret += [[arg.name, value] for key, value in args.items() if arg == key]

            return ret

        return []

    def get_JSON_dict(self):

        json_dict = {
            "class_name": self.__module__.split(".")[-1],
            "input_shape": self.input_shape,
            "mutation_funcs": self.mutation_funcs,
            "args": self._args_to_JSON(),
        }

        ib_json = []
        for block in self.input_blocks:
            ib_json.append(block.toJSON())

        mb_json = []
        for block in self.middle_blocks:
            mb_json.append(block.toJSON())

        ob_json = []
        for block in self.output_blocks:
            ob_json.append(block.toJSON())

        json_dict["input_blocks"] = ib_json
        json_dict["middle_blocks"] = mb_json
        json_dict["output_blocks"] = ob_json

        return json_dict

    def subclass_get_JSON(self, json_dict):

        ignored_args = [
            "parent_block",
            "opt",
            "model",
            "args_enum",
            "inputshape",
            "outputshape",
            "mutations",
            "optimization_goal",
        ]

        for key in self.__dict__.keys():
            if key not in json_dict.keys():
                if (
                    key not in ignored_args
                ):  # We want to ignore these memory references as BA will be reconstructed
                    json_dict[key] = self.__dict__[key]

        return json_dict

    def toJSON(self):

        json_dict = self.get_JSON_dict()

        json_dict = self.subclass_get_JSON(json_dict)

        return json_dict

    @classmethod
    def _get_module(cls):
        return cls.__module__

    @classmethod
    def _get_m_name(cls):
        ret = re.findall(r"^(.*)\.([a-zA-Z0-9]*$)", cls._get_module())
        if len(ret):
            return ret[0]
        else:
            return None

    @classmethod
    def _get_parent_module(cls):
        ret = cls._get_m_name()
        if ret:
            if len(ret) >= 2:
                return ret[0]

    def get_args_enum(self):
        return self.args_enum

    @classmethod
    def _get_args_enum(cls):
        from importlib import import_module

        try:
            args = import_module(cls._get_module()).Args
            return args
        except Exception:
            try:
                args = import_module(cls._get_parent_module()).Args
                return args
            except Exception as e:
                return None


from TensorNAS.Core.LayerMutations import layer_mutation


class Block(BaseBlock):
    @layer_mutation
    def _mutate_drop_subblock(self, verbose=False):
        """
        Randomly drops a middle sub-block
        """
        if len(self.middle_blocks):
            choice_index = random.choice(range(len(self.middle_blocks)))
            block_type = type(self.middle_blocks[choice_index])
            del self.middle_blocks[choice_index]
            ret = "Removed middle block #{} of type {}".format(choice_index, block_type)
            if verbose == True:
                print(ret)
            self.reset_ba_input_shapes()
            return ret

        return "No middle blocks to drop"

    @layer_mutation
    def _mutate_add_subblock(self, verbose=False):
        """
        Randomly adds a sub-block from the provided list of valid sub-blocks
        """
        if len(self.middle_blocks):
            index = random.choice(range(len(self.middle_blocks) + 1))
        else:
            index = 0

        if index > 0:
            input_shape = self.middle_blocks[index - 1].get_output_shape()
        else:
            input_shape = self.input_shape

        new_block_class = self._get_random_subblock_class()
        new_blocks = self.generate_random_sub_block(input_shape, new_block_class)

        if len(new_blocks):
            self.middle_blocks.insert(index, new_blocks[0])

        ret = "Inserted a block of type: {} at index {}".format(new_block_class, index)

        if verbose == True:
            print(ret)

        self.reset_ba_input_shapes()

        return ret
