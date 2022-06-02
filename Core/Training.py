def lrscheduler_decay():
    from Demos import get_global

    def lr_schedule(epoch):
        lrate = get_global("initial_learning_rate") * (
            get_global("decay_per_epoch") ** epoch
        )
        print("Learning rate = {}".format(lrate))
        return lrate

    from tensorflow.keras.callbacks import LearningRateScheduler

    return LearningRateScheduler(lr_schedule)


def lrscheduler_step_function():
    def lr_schedule(epoch):
        if epoch < 12:
            return 0.0005
        elif epoch < 24:
            return 0.0001
        elif epoch < 36:
            return 0.00002
        else:
            return 0.00001

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
