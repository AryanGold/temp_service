import conf
from . import DataPrepare

import time
import logging
import multiprocessing
from queue import Empty, Full # Use standard queue exceptions

from data_providers.ThetaData import ThetaDataProvider
from models.ModelBase import load_model_instance # Use the loader
from ClientDataManager import client_data_manager # Use the shared instance
from process_base import PipelineProcess # Import base class

# Receives data and pushes it into the emitter queue
class ReceiverProcess(PipelineProcess):
    def __init__(self, symbol, model_name, model_settings, provider_config, results_queue, stop_event, command_queue, control_event):
        super().__init__(symbol, model_name, stop_event, command_queue)
        self.model_settings = model_settings
        self.provider_config = provider_config
        self.results_queue = results_queue # Queue to send results to Emitter
        self.control_event = control_event # Separate event for pause/resume state

        # Process-local state
        self.data_provider = None
        self.model_instance = None
        self.request_interval_sec = 1.0 # How often to fetch quotes

    def initialize(self):
        self.logger.info("Initializing DataReceiver...")
        try:
            # Instantiate provider within the process
            self.data_provider = ThetaDataProvider(self.provider_config)
             # Load model instance within the process
            self.model_instance = load_model_instance(self.model_name, self.model_settings)
            self.control_event.set() # Start in running state
            self.logger.info("DataReceiver initialized successfully.")
        except Exception as e:
             self.logger.exception("Initialization failed.")
             raise # Re-raise to stop process startup if essential components fail

    def handle_command(self, command: str, data: any):
        super().handle_command(command, data) # Log command
        if command == "pause":
            self.logger.info("Pausing quote fetching and evaluation.")
            self.control_event.clear()
        elif command == "resume":
            self.logger.info("Resuming quote fetching and evaluation.")
            self.control_event.set()
        elif command == "update_settings":
            self.logger.info(f"Received settings update: {data}")
            # Option 1: Re-initialize model with new settings (simpler)
            try:
                 self.model_settings = data # Store new base settings
                 self.model_instance = load_model_instance(self.model_name, self.model_settings)
                 # Also update shared cache if other processes need latest settings?
                 client_data_manager.update_settings(self.symbol, self.model_name, data)
                 self.logger.info("Model re-initialized with updated settings.")
            except Exception as e:
                 self.logger.exception("Failed to apply settings update by re-initializing model.")


    def work_cycle(self):
        """Fetches quote, gets calibration, evaluates, puts result in queue."""
        start_time = time.monotonic()

        # Wait if paused
        if not self.control_event.wait(timeout=0.1): # Short timeout to remain responsive
            # self.logger.debug("Receiver paused.")
            return # Skip cycle if paused

        try:
            # 1. Get quote (using blocking/sync call within process is fine)
            # Assuming provider methods are synchronous or handle their own async internally
            # If provider IS async, need an event loop WITHIN this process (more complex)
            # For simplicity, assuming sync provider here.
            # quote_data = await self.data_provider.get_quote(self.symbol) # If provider was async
            quote_data = self.data_provider.get_quote_sync(self.symbol) # Assuming a sync version exists
            if quote_data is None:
                self.logger.warning("Failed to get quote data, skipping evaluation.")
                time.sleep(self.request_interval_sec) # Wait before retry on failure
                return

            # 2. Get calibration data from shared manager
            calibration_data = client_data_manager.get_calibration_data(self.symbol, self.model_name)
            # Model evaluation method should handle None calibration_data gracefully

            # 3. Evaluate model (call the external model's method)
            # Assuming SSVI model has an evaluate method like this:
            evaluation_result = self.model_instance.evaluate_model_for_chain(quote_data, calibration_data)

            # 4. Put results into the queue for the Emitter process
            if evaluation_result:
                try:
                    # Use timeout to prevent blocking indefinitely if emitter queue is full
                    self.results_queue.put(evaluation_result, timeout=1.0)
                    # self.logger.debug("Put evaluation result onto results queue.")
                except Full:
                     self.logger.warning("Results queue is full. Discarding evaluation result.")
                except Exception as e:
                     self.logger.exception("Error putting result onto results queue.")
            # else: # Don't log every time evaluation yields nothing
                 # self.logger.debug("No evaluation result produced.")

        except Exception as e:
            self.logger.exception("Error during receiver work cycle.")
            # Avoid spamming logs on repeated errors
            time.sleep(max(self.request_interval_sec, 2.0))

        # Maintain approximate interval
        elapsed = time.monotonic() - start_time
        sleep_duration = max(0, self.request_interval_sec - elapsed)
        time.sleep(sleep_duration)


    def cleanup(self):
        self.logger.info("Cleaning up DataReceiver resources...")
        # Close provider connections if necessary
        if hasattr(self.data_provider, 'disconnect_sync'):
             try:
                 self.data_provider.disconnect_sync()
             except Exception as e:
                 self.logger.error(f"Error disconnecting data provider: {e}")
        self.logger.info("DataReceiver cleanup finished.")
