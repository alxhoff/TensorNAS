class TensorNASIndividual:
    """
    This class is a wrapper object that allows for DEAP individuals to contain the abstract classes defined by TensorNAS Framework
    as DEAP does not allow the use of abstract classes for individuals. As such this class stores a block architecture,
    which is sub-classed from the abstract Block class, and calls into the block architecture to implement DEAP
    functionality.

    Care should be taken to stick with DEAP's requirements, especially the implementation and return values from
    the operations required for DEAP. Eg. DEAP's mutation operation requires that the newly mutated Individual
    be returned from the mutate function, it is not ok to simply mutate the individual as this will break DEAP's
    internal lists for carrying out the EA process.
    """

    def __init__(self, block_architecture):
        self.block_architecture = next(block_architecture)
        self.index = None

    def mutate(self, verbose=False):
        self.index = None
        self.block_architecture.mutate(verbose=verbose)
        return self

    def evaluate(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs,
        batch_size,
        optimizer,
        loss,
        metrics,
        test_name=None,
        model_name=None,
        use_GPU=False,
        q_aware=False,
        logger=None,
    ):
        (param_count, accuracy,) = self.block_architecture.evaluate(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            test_name=test_name,
            model_name=model_name,
            use_GPU=use_GPU,
            q_aware=q_aware,
            logger=logger,
        )
        return param_count, accuracy

    def print(self):
        self.block_architecture.print()

    def __str__(self):
        return str(self.block_architecture)

    def print_summary(self):
        self.block_architecture.print_summary()

    def print_tree(self):
        print(self.block_architecture.get_ascii_tree())
