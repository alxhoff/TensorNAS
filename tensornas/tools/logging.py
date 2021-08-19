from multiprocessing import Process, Manager


def writer(pqueue, filename):

    import os, errno

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, "w+") as log_file:

        while True:
            if pqueue:
                msg = pqueue.get()
                if msg == "STOP":
                    break
                log_file.write("{}\n".format(msg))
                log_file.flush()


class Logger:
    def __init__(self, test_name, subdir=""):
        if subdir != "":
            f_subdir = "/" + subdir
        else:
            f_subdir = subdir
        filename = "Output/{}/{}Logs/tensornas_{}.log".format(
            test_name, f_subdir, test_name + subdir
        )

        m = Manager()
        self.queue = m.Queue()

        reader = Process(
            target=writer,
            args=(
                (
                    self.queue,
                    filename,
                )
            ),
        )
        reader.daemon = True
        reader.start()

    def log(self, string):
        self.queue.put(string)
