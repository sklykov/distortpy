# -*- coding: utf-8 -*-
"""
Multiprocess client: handling the several Processes for computation.

@author: Sergei Klykov, @year: 2023 \n
@licence: MIT \n

"""
# %% Global imports
from multiprocessing import Process, Queue
# multiprocessing.Event provides single boolean flag shared between processes - it's robust but simple
import time
import os
from threading import Thread
from queue import Full, Empty
import numpy as np


# %% Dispatching class
class DispatchImg():

    def __init__(self, finish_timeout_sec: float = 2.5):

        # Initializing class variables
        self.initialized = True; self.close_flag = False
        self.step_delay_ms = 50; self.calculation_started = False
        self.timeout_ms = 1000.0*finish_timeout_sec; self.workers_queues = []

        # Initialize workers
        if os.cpu_count() // 2 > 2:
            self.n_workers = os.cpu_count() // 2
        else:
            self.n_workers = 2
        for n in range(self.n_workers):
            data_queue = Queue(maxsize=10); commands_queue = Queue(maxsize=10)
            worker = MultiInterpolation(commands_queue, data_queue)
            worker.start()
            self.workers_queues.append((worker, commands_queue, data_queue))
        print("# of available CPUs:", os.cpu_count(), "| # of initialized Processes:", self.n_workers)

        # Final initialization of Thread for keeping this program in loop while timeout is not reached
        self.tick_clock = time.perf_counter()  # for tracking usage of the initialized workers
        self.main_thread = Thread(daemon=True, target=self._main_loop)  # note that the daemon thread initialized and run
        self.main_thread.start()  # the daemon thread is checking usage of calculation and clear opened processes if they are not used

    def calculate_interpolation(self, image, zero_ii, zero_jj):
        self.calculation_started = True
        h, w = image.shape
        if h >= w:
            size = h
        else:
            size = w
        patch_size = size // self.n_workers
        if h % self.n_workers != 0:
            last_patch_size = size - (self.n_workers-1)*patch_size
        else:
            last_patch_size = patch_size
        print(patch_size, last_patch_size, size, patch_size*(self.n_workers-1) + last_patch_size)
        self.calculation_started = False

    def _main_loop(self):
        while not self.close_flag:  # Main loop checking the idle time of the calculations
            # Update the latest time point measurment
            if self.calculation_started:
                self.tick_clock = time.perf_counter(); self.calculation_started = False
            # Check if the module used this class is already finished but without calling explicitly close() function
            if round(1000*(time.perf_counter() - self.tick_clock), 0) <= self.timeout_ms:
                time.sleep(self.step_delay_ms/1000)  # sleep and proceed to the next loop step
            else:
                self.close_flag = True
                print("Dispatch calculation finished")

    def close(self):
        if not self.close_flag:
            self.close_flag = True
            # Notify all workers about stopping of operation
            for worker_attributes in self.workers_queues:
                worker, commands_queue, data = worker_attributes
                if not commands_queue.full():
                    try:
                        commands_queue.put_nowait("Stop")
                    except Full:
                        commands_queue.get_nowait()
                        commands_queue.put_nowait("Stop")
                else:
                    commands_queue.get_nowait()
                    try:
                        commands_queue.put_nowait("Stop")
                    except Full:
                        commands_queue.get_nowait()
                        commands_queue.put_nowait("Stop")
            time.sleep(0.2)  # 200 ms pause for possible stopping of all workers
            # Wait that all workers are stopped
            for worker_attributes in self.workers_queues:
                worker, commands_queue, data = worker_attributes
                if worker.is_alive():
                    worker.join(timeout=0.1)
            # Wait the main Thread associated with this class to stop
            if self.main_thread.is_alive():
                self.main_thread.join(timeout=self.timeout_ms/1000)
        print("All workers are realised")


# %% Processing class
class MultiInterpolation(Process):

    def __init__(self, commands_queue: Queue, data_queue: Queue):
        Process.__init__(self)  # use the superclass constructor
        self.commands_queue = commands_queue; self.data_queue = data_queue
        self.running = False; self.main_loop_sleep_ms = 25/1000

    def run(self):
        print("MultiInterpolation class started as Process", flush=True)
        self.running = True
        while self.running:
            if not self.commands_queue.empty():
                try:
                    command = self.commands_queue.get_nowait()
                    # Handling commands as strings
                    if isinstance(command, str):
                        if command == "Stop":
                            self.running = False; break
                        else:
                            print("Received unrecognized command: ", command, flush=True)
                    # Handling incoming Exceptopns
                    elif isinstance(command, Exception):
                        self.running = False; break
                except Empty:
                    time.sleep(self.main_loop_sleep_ms)
            else:
                time.sleep(self.main_loop_sleep_ms)
        print("MultiInterpolation Process finished", flush=True)


# %% Testing functionality
if __name__ == "__main__":
    di = DispatchImg(); img = np.zeros((570, 531))
    di.calculate_interpolation(img, None, None)
    time.sleep(1.0); di.close()
