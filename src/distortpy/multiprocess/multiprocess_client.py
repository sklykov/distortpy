# -*- coding: utf-8 -*-
"""
Multiprocess client: handling the several Processes for computation.

@author: Sergei Klykov, @year: 2023 \n
@licence: MIT \n

"""
# %% Global imports
from multiprocessing import Process, Queue
import time
import os
from threading import Thread


# %% Dispatching class
class DispatchImg():

    def __init__(self, finish_timeout_sec: float = 2.5):

        # Initializing class variables
        self.initialized = True; self.close_flag = False
        self.step_delay_ms = 50; self.calculation_started = False
        self.timeout_ms = 1000.0*finish_timeout_sec

        # Initialize workers
        # print("# of available CPUs:", os.cpu_count())
        if os.cpu_count() // 2 > 2:
            n_workers = os.cpu_count() // 2
        else:
            n_workers = 2
        self.workers = []; self.commands_queues = []; self.data_queues = []
        for i in range(n_workers):
            pass

        self.tick_clock = time.perf_counter()  # for tracking usage of the initialized workers
        self.main_thread = Thread(daemon=True, target=self._main_loop)  # note that the daemon thread initialized and run
        self.main_thread.start()  # the daemon thread is checking usage of calculation and clear opened processes if they are not used

    def calculate(self):
        self.calculation_started = True
        self.calculation_started = False

    def _main_loop(self):
        while not self.close_flag:  # Main loop checking the idle time of the calculations
            if self.calculation_started:
                self.tick_clock = time.perf_counter(); self.calculation_started = False
            if round(1000*(time.perf_counter() - self.tick_clock), 0) <= self.timeout_ms:
                time.sleep(self.step_delay_ms/1000)  # sleep and proceed to the next loop step
            else:
                self.close_flag = True
                print("Dispatch calculation finished")

    def close(self):
        if not self.close_flag:
            self.close_flag = True
            if self.main_thread.is_alive():
                self.main_thread.join(timeout=self.timeout_ms/1000)
        print("All workers are realised")


# %% Processing class
class MultiInterpolation(Process):

    def __init__(self, commands_queue: Queue, data_queue: Queue):
        self.commands_queue = commands_queue; self.data_queue = data_queue

    def run(self):
        pass


# %% Testing functionality
if __name__ == "__main__":
    di = DispatchImg(); time.sleep(1.0); di.close()
    print("Check")
