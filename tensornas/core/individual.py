class Individual:
    def __init__(self, block_architecture):
        self.block_architecture = next(block_architecture)

    def mutate(self):
        self.block_architecture.mutate()

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
    ):
        self.block_architecture.evaluate(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
