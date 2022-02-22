from time import time, strftime, localtime


class Timer():
    """Simple class to track execution time of scripts"""

    def __init__(self):
        self.start = time()
        self.elapsed = None
        self.checkpoint_time = None
        self.log("Start Program")

    def log(self, s=""):
        line = "="*40
        print(line)
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()), '-', s)
        if self.elapsed:
            print("Elapsed time:", self.elapsed)
        print(line)
        print()

    def checkpoint(self, s=""):
        if not self.checkpoint_time:
            self.checkpoint_time = time()
            elapsed_beginning = elapsed_since_last_checkpoint = self.checkpoint_time - self.start
        else:
            tmp_checkpoint_time = self.checkpoint_time
            self.checkpoint_time = time()
            elapsed_since_last_checkpoint = self.checkpoint_time - tmp_checkpoint_time
            elapsed_beginning = self.checkpoint_time - self.start

        line = "="*40
        print(line)
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()), '-', s)
        print("Elapsed since begin:", round(elapsed_beginning, 10))
        print("Elapsed since last checkpoint:", round(elapsed_since_last_checkpoint, 10))
        print(line)
        print()

    def end(self):
        self.end = time()
        self.elapsed = self.end-self.start
        self.checkpoint("End Program")
