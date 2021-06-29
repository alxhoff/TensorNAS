from multiprocessing import Process, Queue
from time import gmtime, strftime


def writer(pqueue, filename):

    with open(filename, "w+") as log_file:

        while True:
            if pqueue:
                msg = pqueue.get()
                if msg == "STOP":
                    break
                log_file.write("{}\n".format(msg))
                log_file.flush()


class Logger:
    def __init__(self):
        filename = "tensornas_{}.log".format(strftime("%Y%m%d-%H%M", gmtime()))

        self.queue = Queue()

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
