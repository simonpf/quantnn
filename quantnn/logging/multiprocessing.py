"""
===============================
quantnn.logging.multiprocessing
===============================

This module defines utility function to handle logging from sub-processes.
"""
import logging
from logging import handlers
import multiprocessing
import threading

_LOG_QUEUE = None


def get_log_queue():
    """
    Return global logging queue.
    """
    global _LOG_QUEUE
    if _LOG_QUEUE is None:
        _LOG_QUEUE = multiprocessing.Queue()
    return _LOG_QUEUE


class SubprocessLogging(multiprocessing.Process):
    """
    Base class to handle logging from subprocesses. Subprocesses should
    inherit from this class in order to have their log messages displayed
    cleanly.
    """
    def __init__(self):
        super().__init__()
        self.log_queue = get_log_queue()

    def run(self):
        import quantnn.logging
        root = logging.getLogger()
        root.handlers = [handlers.QueueHandler(self.log_queue)]


class LoggingThread(threading.Thread):
    """
    Thread to log messages from working processes.
    """
    def __init__(self,
                 log_queue):
        """
        Args:
            log_queue: The queue to which other processes are logging their
                messages.

        """
        super().__init__()
        self.log_queue = log_queue

    def run(self):
        """
        Listen on queue and print incoming messages.
        """
        while True:
            record = self.log_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)


_USERS = 0
_LOGGING_THREAD = None


def start_logging():
    """
    Starts the listener thread that prints messages from the subprocesses.
    """
    global _USERS, _LOGGING_THREAD
    _USERS += 1
    if _LOGGING_THREAD is None:
        _LOGGING_THREAD = LoggingThread(get_log_queue())
        _LOGGING_THREAD.start()


def stop_logging():
    """
    Signal that no more messages are expected from subprocesses.
    """
    global _USERS, _LOGGING_THREAD
    _USERS -= 1
    if _USERS <= 0:
        if _LOGGING_THREAD is not None:
            get_log_queue().put(None)
            _LOGGING_THREAD.join()
            _LOGGING_THREAD = None
