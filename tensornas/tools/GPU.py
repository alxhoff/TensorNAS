import tensorflow as tf


def Enable():
    ### ENABLE GPU ###
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    ##################
