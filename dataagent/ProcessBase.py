import multiprocessing
import logging
import time
from abc import ABC, abstractmethod

# Configure logging for the process (can be refined with QueueHandler)
def setup_process_logging():
     # Basic configuration for each process log - consider QueueHandler for centralized logging
     logging.basicConfig(
         level=logging.DEBUG, # Or get from config
         format=f'%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
         datefmt='%Y-%m-%d %H:%M:%S'
     )

class PipelineProcess(multiprocessing.Process, ABC):
    def __init__(self, symbol, model_name, stop_event, command_queue):
        super().__init__(name=f"{self.__class__.__name__}-{symbol}-{model_name}")
        self.symbol = symbol
        self.model_name = model_name
        self.stop_event = stop_event # Event to signal process termination
        self.command_queue = command_queue # Queue for receiving commands (pause, resume, update)
        self.logger = None # Logger will be configured in run()

    def run(self):
        """Main process entry point."""
        setup_process_logging()
        self.logger = logging.getLogger(self.name)
        self.logger.info("Process started.")
        try:
            self.initialize()
            while not self.stop_event.is_set():
                # Check for commands without blocking indefinitely
                try:
                    command, data = self.command_queue.get_nowait()
                    self.handle_command(command, data)
                except multiprocessing.queues.Empty:
                    pass # No command waiting

                # Perform main work cycle
                self.work_cycle()

                # Prevent busy-waiting if work_cycle is very fast or has internal waits
                # time.sleep(0.01) # Small sleep if necessary

        except Exception as e:
            self.logger.exception("Unhandled exception in process run loop.")
        finally:
            self.cleanup()
            self.logger.info("Process terminating.")

    @abstractmethod
    def initialize(self):
        """Setup resources needed by the process."""
        pass

    @abstractmethod
    def handle_command(self, command: str, data: any):
        """Process commands received from the main service."""
        self.logger.debug(f"Received command: {command}, Data: {data}")
        if command == "stop": # Explicit stop command
             self.stop_event.set()
        # Subclasses should handle pause, resume, update etc.

    @abstractmethod
    def work_cycle(self):
        """The main logic executed repeatedly in the process loop."""
        pass

    @abstractmethod
    def cleanup(self):
        """Release resources before the process exits."""
        pass
        