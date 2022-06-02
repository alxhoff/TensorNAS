def config_GPU():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.compat.v1.Session(config=config)

    return
