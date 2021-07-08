def config_GPU():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
