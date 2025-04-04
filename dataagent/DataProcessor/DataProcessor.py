import time
import logging
import multiprocessing
from queue import Empty

from models.ModelBase import load_model_instance
from ClientDataManager import client_data_manager
from process_base import PipelineProcess
from conf import DEFAULT_CALIBRATION_INTERVAL_SECONDS

# Processes data for a specific client symbol/model pair
class ProcessorProcess(PipelineProcess):
    def __init__(self, symbol, model_name, model_settings, stop_event, command_queue, interval_seconds=None):
        super().__init__(symbol, model_name, stop_event, command_queue)
        self.model_settings = model_settings
        self.interval_seconds = interval_seconds if interval_seconds is not None else DEFAULT_CALIBRATION_INTERVAL_SECONDS

        # Process-local state
        self.model_instance = None
        self._last_calibration_time = 0
        self._force_recalibrate = False

    def initialize(self):
        self.logger.info("Initializing DataProcessor...")
        try:
             self.model_instance = load_model_instance(self.model_name, self.model_settings)
             # Run initial calibration immediately?
             # self.logger.info("Running initial calibration...")
             # self._run_calibration()
             # self._last_calibration_time = time.time()
             self.logger.info("DataProcessor initialized successfully.")
        except Exception as e:
             self.logger.exception("Initialization failed.")
             raise

    def handle_command(self, command: str, data: any):
        super().handle_command(command, data)
        if command == "force_recalibrate":
             self.logger.info("Received command to force recalibration.")
             self._force_recalibrate = True
        elif command == "update_settings":
             self.logger.info(f"Received settings update: {data}")
             # Similar to Receiver, either re-initialize or call an update method
             try:
                 self.model_settings = data
                 self.model_instance = load_model_instance(self.model_name, self.model_settings)
                 # Trigger recalibration after settings change?
                 self._force_recalibrate = True
                 self.logger.info("Model re-initialized with updated settings. Recalibration triggered.")
             except Exception as e:
                 self.logger.exception("Failed to apply settings update by re-initializing model.")


    def work_cycle(self):
        """Checks if it's time to recalibrate and runs it."""
        now = time.time()
        time_since_last = now - self._last_calibration_time

        if self._force_recalibrate or (self.interval_seconds > 0 and time_since_last >= self.interval_seconds) :
            self.logger.info(f"Triggering calibration. Forced: {self._force_recalibrate}, Interval Met: {time_since_last >= self.interval_seconds}")
            try:
                self._run_calibration()
                self._last_calibration_time = time.time() # Update time only on success
                self._force_recalibrate = False # Reset flag
            except Exception as e:
                self.logger.exception("Calibration failed.")
                # Don't update last calibration time on failure, maybe retry sooner?

        # Sleep for a short duration to avoid pegging CPU checking the time/commands.
        # Also allows time for commands to arrive.
        # Calculate sleep time to approximate next interval check? More complex.
        # Simpler: Sleep for a fixed short interval.
        check_interval = min(self.interval_seconds / 10, 1.0) if self.interval_seconds > 0 else 1.0
        time.sleep(check_interval)


    def _run_calibration(self):
        """Executes the model's calibration method and updates shared cache."""
        self.logger.info("Starting model calibration...")
        start_time = time.time()
        try:
            # Assuming SSVI model has a calibrate method like this:
            new_calibration_data = self.model_instance.calibrate_model_params_to_chain()

            duration = time.time() - start_time
            if new_calibration_data:
                # Add metadata if needed
                new_calibration_data['calibrated_at_epoch'] = time.time()
                new_calibration_data['calibration_duration_sec'] = round(duration, 2)

                # Update the shared cache via ClientDataManager
                client_data_manager.update_calibration_data(self.symbol, self.model_name, new_calibration_data)
                self.logger.info(f"Calibration successful in {duration:.2f} seconds. Cache updated.")
            else:
                self.logger.warning(f"Calibration completed in {duration:.2f} seconds but returned no data.")

        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Error during calibration (duration: {duration:.2f}s).")
            raise # Re-raise so the main loop catches it

    def cleanup(self):
        self.logger.info("Cleaning up DataProcessor resources...")
        # No specific resources assumed here, add if needed
        self.logger.info("DataProcessor cleanup finished.")