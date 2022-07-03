def lrscheduler_decay():
    from Demos import get_global
    from tensorflow.keras.callbacks import LearningRateScheduler

    def lr_schedule(epoch):
        lrate = get_global("initial_learning_rate") * (
            get_global("decay_per_epoch") ** epoch
        )
        print("Learning rate = %f" % lrate)
        return lrate

    return LearningRateScheduler(lr_schedule)


def lrscheduler_step_function():
    def lr_schedule(epoch):
        lr = 0.00001
        if epoch < 12:
            lr = 0.0005
        elif epoch < 24:
            lr = 0.0001
        elif epoch < 36:
            lr = 0.00002
        else:
            lr = 0.00001

        print("Learning rate = %f" % lr)
        return lr

    from tensorflow.keras.callbacks import LearningRateScheduler

    return LearningRateScheduler(lr_schedule)


def get_early_stopper():
    from tensorflow.keras.callbacks import EarlyStopping
    from Demos import get_global

    monitor = get_global("stopper_monitor")
    patience = get_global("patience")
    min_delta = get_global("stopper_min_delta")
    mode = get_global("stopper_mode")

    return EarlyStopping(
        monitor=monitor, patience=patience, min_delta=min_delta, mode=mode
    )
