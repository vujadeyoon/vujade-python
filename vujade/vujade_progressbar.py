"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_progressbar.py
Description: A module for progressbar
"""


import progressbar


class ProgressBar(object):
    def __init__(self, num_total: int) -> None:
        super(ProgressBar, self).__init__()
        self.cnt = 0
        self.bar = progressbar.ProgressBar(maxval=num_total).start()

    def update(self):
        self.cnt += 1
        self.bar.update(self.cnt)

    def get(self):
        return self.bar

    def finish(self):
        self.bar.finish()
