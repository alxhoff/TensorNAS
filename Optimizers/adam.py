import random

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BETA_1 = 0.9
DEFAULT_BETA_2 = 0.999
DEFAULT_EPSILON = 1e-07


class Optimizer:
    def __init__(self):

        self.learning_rate = DEFAULT_LEARNING_RATE
        self.beta_1 = DEFAULT_BETA_1
        self.beta_2 = DEFAULT_BETA_2
        self.epsilon = DEFAULT_EPSILON

    def get_optimizer(self):

        import tensorflow as tf

        opt = tf.keras.optimizers.Adam(
            learning_rate=DEFAULT_LEARNING_RATE,
            beta_1=DEFAULT_BETA_1,
            beta_2=DEFAULT_BETA_2,
            epsilon=DEFAULT_EPSILON,
        )

        return opt

    def mutate(self, verbose=False):
        from TensorNAS.Core.Mutate import mutate_float

        choice = random.randrange(0, 5)

        if choice == 0:
            if verbose:
                print("[MUTATE] mutating learning rate")
            self.learning_rate = mutate_float(self.learning_rate, 0, 1)
        elif choice == 1:
            if verbose:
                print("[MUTATE] mutating beta 1")
            self.beta_1 = mutate_float(self.beta_1, 0, 1)
        elif choice == 2:
            if verbose:
                print("[MUTATE] mutating beta 2")
            self.beta_2 = mutate_float(self.beta_2, 0, 1)
        elif choice == 3:
            if verbose:
                print("[MUTATE] mutating epsilon")
            from TensorNAS.Core.Mutate import MutationOperators

            self.epsilon = mutate_float(
                self.epsilon,
                0,
                1,
                operator=MutationOperators.STEP_DOWN,
                step_size=0.000000001,
            )
