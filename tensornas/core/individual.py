class Individual:
    def __init__(self, block_architecture):
        self.block_architecture = next(block_architecture)

    def mutate(self):
        self.block_architecture.mutate()

    def evaluate(self):
        self.block_architecture.evaluate()
