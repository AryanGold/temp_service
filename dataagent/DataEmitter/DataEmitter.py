import time
import logging
import multiprocessing
from queue import Empty, Full
import json

from process_base import PipelineProcess

# Sends processed data via a pipe to the main WebSocket process.
class EmitterProcess(PipelineProcess):
    def __init__(self, symbol, model_name, results_queue, websocket_queue, stop_event, command_queue):
        super().__init__(symbol, model_name, stop_event, command_queue)
        self.results_queue = results_queue # Queue to receive results from Receiver
        self.websocket_queue = websocket_queue # Queue to send formatted messages TO main service

    def initialize(self):
        self.logger.info("Initializing DataEmitter...")
        # No specific resources needed usually, unless formatting requires complex setup
        self.logger.info("DataEmitter initialized successfully.")

    def handle_command(self, command: str, data: any):
        super().handle_command(command, data)
        # Emitter might not need specific commands other than 'stop'

    def work_cycle(self):
        """Gets result from queue, formats, puts message in websocket queue."""
        try:
            # Get result from receiver queue, wait up to 1 sec
            result_payload = self.results_queue.get(timeout=1.0)
            # self.logger.debug(f"Got result from receiver queue: {result_payload}")

            # Format the message for WebSocket client
            ws_message = {
                "type": "data_stream",
                "symbol": self.symbol,
                "model": self.model_name,
                "payload": result_payload
            }

            # Put the formatted message onto the queue for the main service
            try:
                # Use timeout to prevent blocking if main service is slow
                self.websocket_queue.put(ws_message, timeout=1.0)
                # self.logger.debug("Put formatted message onto websocket queue.")
            except Full:
                 self.logger.warning("WebSocket queue is full. Discarding message.")
            except Exception as e:
                 self.logger.exception("Error putting message onto websocket queue.")

        except Empty:
            # self.logger.debug("No results in queue, waiting...")
            # No result waiting, loop will continue after short sleep (handled by ProcessBase/run loop)
            pass
        except Exception as e:
            self.logger.exception("Error during emitter work cycle.")
            # Avoid spamming logs on repeated errors
            time.sleep(1.0)


    def cleanup(self):
        self.logger.info("Cleaning up DataEmitter resources...")
        # No specific resources assumed here
        self.logger.info("DataEmitter cleanup finished.")
