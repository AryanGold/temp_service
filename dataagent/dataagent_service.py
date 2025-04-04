# dataagent/dataagent_service.py
import asyncio
import json
import logging
import signal
import uuid
from pathlib import Path
from typing import Dict, Tuple, Any
import multiprocessing # Main process uses this

import websockets
import yaml

# Import process classes and shared components
from conf import glob_manager # We need to manage its lifecycle
from ClientDataManager import ClientDataManager
from models.ModelBase import load_model_instance # Only needed for validation here?
from data_receiver import ReceiverProcess
from data_processor import ProcessorProcess
from data_emitter import EmitterProcess

# Import provider config loader if separate (or handle within service)
# from data_providers.ThetaData import load_data_provider_config # Example

logger = logging.getLogger(__name__)

# Type alias for pipeline key (websocket is no longer part of the key for process mgmt)
# We map websocket connection to the pipeline structure managing its processes
PipelineMgmtKey = Tuple[str, str] # (symbol, model_name)

# Structure to hold pipeline process info associated with a websocket connection
class ClientPipelineInfo:
    def __init__(self, symbol: str, model_name: str, receiver_proc: ReceiverProcess, processor_proc: ProcessorProcess, emitter_proc: EmitterProcess, command_queue: multiprocessing.Queue, websocket_queue: multiprocessing.Queue, control_event: multiprocessing.Event, stop_event: multiprocessing.Event):
        self.symbol = symbol
        self.model_name = model_name
        self.receiver_proc = receiver_proc
        self.processor_proc = processor_proc
        self.emitter_proc = emitter_proc
        self.command_queue = command_queue
        self.websocket_queue = websocket_queue
        self.control_event = control_event # For receiver pause/resume
        self.stop_event = stop_event       # For signaling process stop
        self.is_running = False

    def start_processes(self):
        self.logger = logging.getLogger(f"Pipeline-{self.symbol}-{self.model_name}") # Logger for mgmt actions
        self.logger.info("Starting pipeline processes...")
        self.processor_proc.start()
        self.receiver_proc.start()
        self.emitter_proc.start()
        self.is_running = True
        self.logger.info("Pipeline processes started.")

    def stop_processes(self):
        self.logger = logging.getLogger(f"Pipeline-{self.symbol}-{self.model_name}")
        if not self.is_running:
            self.logger.info("Pipeline processes already stopped or not started.")
            return

        self.logger.info("Stopping pipeline processes...")
        self.is_running = False # Mark as stopping
        # Signal processes to stop
        self.stop_event.set()
        # Send explicit stop command in case event isn't checked immediately
        try:
             self.command_queue.put_nowait(("stop", None))
        except multiprocessing.queues.Full:
             self.logger.warning("Command queue full while trying to send stop signal.")

        # Wait for processes to terminate
        processes = [self.emitter_proc, self.receiver_proc, self.processor_proc]
        for proc in processes:
             try:
                 proc.join(timeout=5.0) # Wait max 5 seconds per process
                 if proc.is_alive():
                      self.logger.warning(f"Process {proc.name} did not terminate gracefully, terminating forcefully.")
                      proc.terminate() # Force terminate if join times out
                      proc.join(timeout=1.0) # Short wait after terminate
                 else:
                      self.logger.info(f"Process {proc.name} terminated gracefully. Exit code: {proc.exitcode}")
             except Exception as e:
                  self.logger.error(f"Error joining/terminating process {proc.name}: {e}")

        # Close queues (optional, helps prevent hangs if processes didn't exit cleanly)
        self.command_queue.close()
        self.websocket_queue.close()
        # Note: Results queue between Receiver/Emitter is managed by them or implicitly closed

        # Clean up shared data
        ClientDataManager.clear_pipeline_data(self.symbol, self.model_name)
        self.logger.info("Pipeline processes stopped and resources cleaned.")

    def send_command(self, command: str, data: Any = None):
        if not self.is_running:
            self.logger.warning(f"Attempted to send command '{command}' to stopped pipeline.")
            return False
        try:
            self.command_queue.put_nowait((command, data))
            self.logger.info(f"Sent command '{command}' to pipeline.")
            return True
        except Full:
            self.logger.error(f"Command queue full for pipeline. Failed to send command '{command}'.")
            return False
        except Exception as e:
             self.logger.exception(f"Error sending command '{command}' to pipeline.")
             return False


class DataAgentService:
    def __init__(self, config_path="service_conf.yaml"):
        self._configure_basic_logging() # Configure early for init logs
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._configure_logging() # Re-configure based on loaded config

        # Load available models {name: default_settings}
        self.available_models: Dict[str, Dict] = self._load_models_from_config()

        # Runtime state
        self.websocket_server = None
        self.shutdown_event = asyncio.Event() # Asyncio event for main service shutdown
        # Maps websocket object to a dict mapping PipelineMgmtKey to ClientPipelineInfo
        self.client_pipelines: Dict[websockets.WebSocketServerProtocol, Dict[PipelineMgmtKey, ClientPipelineInfo]] = {}
        # Maps websocket object to the listener task handling its websocket_queue
        self.client_listeners: Dict[websockets.WebSocketServerProtocol, asyncio.Task] = {}

        # Ensure global manager is available
        if glob_manager is None:
            raise RuntimeError("Failed to initialize multiprocessing Manager in conf.py")

        logger.info("DataAgentService initialized.")
        logger.info(f"Available models: {list(self.available_models.keys())}")


    def _configure_basic_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def _load_config(self):
        # ... (same as previous version)
        if not self.config_path.is_file():
             logger.error(f"Configuration file not found: {self.config_path}")
             raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
             try:
                 config = yaml.safe_load(f)
                 logger.info(f"Configuration loaded from {self.config_path}")
                 return config
             except yaml.YAMLError as e:
                 logger.error(f"Error parsing configuration file: {e}")
                 raise ValueError(f"Invalid YAML configuration: {e}")

    def _configure_logging(self):
        # ... (same as previous version)
        log_level_str = self.config.get('logging', {}).get('level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        # Consider adding process name to format for clarity
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(processName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True # Override basicConfig from _configure_basic_logging
        )
        logging.getLogger("websockets").setLevel(logging.INFO)
        logger.info(f"Logging configured to level {log_level_str}")


    def _load_models_from_config(self) -> Dict[str, Dict]:
         # ... (same as previous version)
         models_conf = self.config.get('models', [])
         if not isinstance(models_conf, list):
              logger.error("Invalid 'models' format in config. Expected a list.")
              return {}

         loaded_models = {}
         for model_spec in models_conf:
             if isinstance(model_spec, dict) and 'name' in model_spec:
                 name = model_spec['name']
                 # Validate model can be loaded here? Optional, but good.
                 try:
                      # Try loading just to see if it exists, don't keep instance
                      _ = load_model_instance(name, {})
                      logger.debug(f"Validated model '{name}' is loadable.")
                 except ValueError as e:
                      logger.warning(f"Model '{name}' specified in config failed validation: {e}. It will be listed but may fail on 'add'.")
                 except Exception as e:
                      logger.warning(f"Unexpected error validating model '{name}': {e}")

                 settings = model_spec.get('settings', {})
                 loaded_models[name] = settings
             else:
                  logger.warning(f"Skipping invalid model specification in config: {model_spec}")
         return loaded_models

    async def _register_client(self, websocket):
        client_id = str(uuid.uuid4().hex[:8])
        self.client_pipelines[websocket] = {} # Init pipeline dict for this client
        self.client_listeners[websocket] = None # Placeholder for listener task
        logger.info(f"Client connected: {websocket.remote_address}, ID: {client_id}")


    async def _unregister_client(self, websocket):
        client_info = self.client_pipelines.pop(websocket, None)
        listener_task = self.client_listeners.pop(websocket, None)

        if client_info is not None:
            client_id = f"{websocket.remote_address}" # Use address for logging id now
            logger.info(f"Client disconnected: {client_id}. Cleaning up resources...")

            # Stop the listener task for this client
            if listener_task and not listener_task.done():
                 logger.info(f"Cancelling listener task for client {client_id}...")
                 listener_task.cancel()
                 try:
                      await asyncio.wait_for(listener_task, timeout=1.0)
                      logger.info(f"Listener task cancelled for client {client_id}.")
                 except asyncio.CancelledError:
                      logger.info(f"Listener task cancellation confirmed for client {client_id}.")
                 except asyncio.TimeoutError:
                      logger.warning(f"Listener task for client {client_id} did not cancel within timeout.")
                 except Exception as e:
                      logger.error(f"Error cancelling listener task for client {client_id}: {e}")


            # Stop all pipeline processes associated with this client
            pipeline_keys = list(client_info.keys())
            if pipeline_keys:
                logger.info(f"Stopping {len(pipeline_keys)} pipelines for disconnected client {client_id}.")
                # Stopping processes involves blocking joins, run in executor
                # to avoid blocking the main async loop.
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._stop_client_pipelines_sync, client_info)
            logger.info(f"Cleanup finished for client {client_id}.")
        else:
             logger.warning(f"Attempted to unregister unknown client: {websocket.remote_address}")

    def _stop_client_pipelines_sync(self, client_pipeline_dict):
         """Synchronous helper to stop all pipelines for a client."""
         for pipeline_info in client_pipeline_dict.values():
             pipeline_info.stop_processes() # This method handles logging and joining

    async def send_json(self, websocket, data: Dict):
        # ... (same as previous version, maybe add check for self.shutdown_event)
        if self.shutdown_event.is_set() or websocket.closed:
            # Don't try to send if shutting down or already closed
            return
        try:
            await websocket.send(json.dumps(data))
        except websockets.ConnectionClosed:
            logger.warning(f"Connection closed during send by peer: {websocket.remote_address}. Triggering cleanup.")
            # Unregister will be called by the main handler's finally block
        except Exception as e:
            logger.error(f"Error sending JSON to {websocket.remote_address}: {e}")


    async def send_error(self, websocket, message: str, request_data: Dict = None):
        # ... (same as previous version)
         error_payload = { "status": "error", "message": message }
         if request_data and isinstance(request_data.get('request_id'), (str, int)):
              error_payload["request_id"] = request_data["request_id"]
         await self.send_json(websocket, error_payload)

    async def send_success(self, websocket, data: Dict, request_data: Dict = None):
         # ... (same as previous version)
         success_payload = { "status": "success", "data": data }
         if request_data and isinstance(request_data.get('request_id'), (str, int)):
              success_payload["request_id"] = request_data["request_id"]
         await self.send_json(websocket, success_payload)


    # --- WebSocket Queue Listener ---
    async def _websocket_emitter_listener(self, websocket, websocket_queue: multiprocessing.Queue):
        """Listens to a queue from an Emitter process and sends data over WebSocket."""
        client_id = f"{websocket.remote_address}"
        logger.info(f"[Listener-{client_id}] Starting listener for WebSocket queue.")
        loop = asyncio.get_running_loop()
        try:
            while not self.shutdown_event.is_set() and not websocket.closed:
                try:
                    # Use run_in_executor to perform the blocking queue get
                    message = await loop.run_in_executor(None, websocket_queue.get, True, 0.5) # Timeout 0.5s
                    if message:
                        await self.send_json(websocket, message)
                except multiprocessing.queues.Empty:
                    # Queue is empty, continue loop check conditions
                    await asyncio.sleep(0.05) # Small sleep to prevent high CPU usage
                except websockets.ConnectionClosed:
                     logger.warning(f"[Listener-{client_id}] WebSocket closed while listener active.")
                     break # Exit loop if connection is closed
                except Exception as e:
                     # Log errors but keep listener running unless it's fatal
                     logger.error(f"[Listener-{client_id}] Error reading from queue or sending to WebSocket: {e}", exc_info=True)
                     await asyncio.sleep(1) # Pause briefly after error
        except asyncio.CancelledError:
             logger.info(f"[Listener-{client_id}] Listener task cancelled.")
        except Exception as e:
             logger.exception(f"[Listener-{client_id}] Unexpected error in listener loop.")
        finally:
             logger.info(f"[Listener-{client_id}] Listener task stopped.")
             # Queue cleanup might be handled elsewhere (e.g., unregister_client)


    # --- Pipeline Management (Process Based) ---

    async def _start_pipeline(self, websocket, symbol: str, model_name: str, settings_override: dict):
        """Creates IPC, launches processes, and stores pipeline info."""
        mgmt_key = (symbol, model_name)
        client_id = f"{websocket.remote_address}"

        if websocket not in self.client_pipelines:
             await self.send_error(websocket, "Client not registered properly.")
             return
        if mgmt_key in self.client_pipelines[websocket]:
            await self.send_error(websocket, f"Pipeline for {symbol}/{model_name} already exists for this client.")
            return

        if model_name not in self.available_models:
            await self.send_error(websocket, f"Model '{model_name}' is not supported.")
            return

        logger.info(f"[Client {client_id}] Request to start pipeline: {symbol}/{model_name}")

        try:
            # 1. Create IPC Objects (before launching processes)
            # Queue for Receiver -> Emitter results
            results_queue = multiprocessing.Queue(maxsize=100) # Buffer results
            # Queue for Emitter -> Service (WebSocket sending)
            websocket_queue = multiprocessing.Queue(maxsize=100)
            # Queue for Service -> Processes commands
            command_queue = multiprocessing.Queue()
            # Event for Receiver pause/resume
            control_event = multiprocessing.Event()
            # Event for signaling processes to stop
            stop_event = multiprocessing.Event()

            # 2. Prepare Configs/Settings
            initial_model_settings = self.available_models[model_name].copy()
            initial_model_settings.update(settings_override)
            # Get provider config (assuming it's simple dict for now)
            provider_config = self.config.get('data_providers', {}).get('ThetaData', {})
            processor_interval = self.config.get('processor', {}).get('calibration_interval_seconds')

            # Store initial settings in shared cache
            ClientDataManager.update_settings(symbol, model_name, initial_model_settings)


            # 3. Create Process Instances
            receiver_proc = ReceiverProcess(
                symbol, model_name, initial_model_settings, provider_config,
                results_queue, stop_event, command_queue, control_event
            )
            processor_proc = ProcessorProcess(
                symbol, model_name, initial_model_settings,
                stop_event, command_queue, processor_interval
            )
            emitter_proc = EmitterProcess(
                symbol, model_name, results_queue, websocket_queue,
                stop_event, command_queue
            )

            # 4. Store Pipeline Info
            pipeline_info = ClientPipelineInfo(
                symbol, model_name, receiver_proc, processor_proc, emitter_proc,
                command_queue, websocket_queue, control_event, stop_event
            )
            self.client_pipelines[websocket][mgmt_key] = pipeline_info

            # 5. Start Processes (handled by ClientPipelineInfo)
            # Running start_processes involves blocking process start, do in executor?
            # For simplicity here, assume start() is fast enough not to block excessively.
            # loop = asyncio.get_running_loop()
            # await loop.run_in_executor(None, pipeline_info.start_processes)
            pipeline_info.start_processes() # Direct call for now

            # 6. Start the listener task for this pipeline's websocket queue
            listener_task = asyncio.create_task(
                 self._websocket_emitter_listener(websocket, websocket_queue),
                 name=f"Listener-{client_id}-{symbol}-{model_name}"
            )
            # Store listener task (or maybe associate with pipeline_info?)
            # Overwrite if exists? For now, store per client - assumes one listener ok.
            # If one listener per pipeline needed, store in pipeline_info.
            if self.client_listeners[websocket] is not None and not self.client_listeners[websocket].done():
                logger.warning(f"[Client {client_id}] Existing listener task found, cancelling before starting new one.")
                self.client_listeners[websocket].cancel() # Cancel previous if any
            self.client_listeners[websocket] = listener_task


            await self.send_success(websocket, {"message": f"Pipeline starting for {symbol}/{model_name}"})
            logger.info(f"[Client {client_id}] Pipeline started: {symbol}/{model_name}")

        except Exception as e:
             logger.error(f"[Client {client_id}] Failed to start pipeline {symbol}/{model_name}: {e}", exc_info=True)
             await self.send_error(websocket, f"Internal server error starting pipeline.")
             # Clean up any partially created resources if possible
             # ... (e.g., close queues, ensure manager data cleared)


    async def _stop_pipeline(self, websocket, symbol: str, model_name: str):
        """Stops processes and cleans up resources for a specific pipeline."""
        mgmt_key = (symbol, model_name)
        client_id = f"{websocket.remote_address}"

        if websocket not in self.client_pipelines or mgmt_key not in self.client_pipelines[websocket]:
             await self.send_error(websocket, f"Pipeline for {symbol}/{model_name} not found for this client.")
             return False

        logger.info(f"[Client {client_id}] Request to stop pipeline: {symbol}/{model_name}")
        pipeline_info = self.client_pipelines[websocket].pop(mgmt_key) # Remove from active dict

        # Stop processes (blocking, run in executor)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline_info.stop_processes)

        # Stop the associated listener task? If listener is per-pipeline.
        # If listener is per-client, only stop on disconnect.

        await self.send_success(websocket, {"message": f"Pipeline stopped for {symbol}/{model_name}"})
        logger.info(f"[Client {client_id}] Pipeline stopped successfully: {symbol}/{model_name}")
        return True


    async def _control_pipeline(self, websocket, symbol: str, model_name: str, action: str, data: Dict = None):
        """Sends a command (pause, resume, update) to a running pipeline."""
        mgmt_key = (symbol, model_name)
        client_id = f"{websocket.remote_address}"

        pipeline_info = self.client_pipelines.get(websocket, {}).get(mgmt_key)

        if not pipeline_info or not pipeline_info.is_running:
            await self.send_error(websocket, f"Pipeline for {symbol}/{model_name} not found or not running.")
            return

        logger.info(f"[Client {client_id}] Request action '{action}' for pipeline: {symbol}/{model_name}")
        command_data = None
        command = action # Default command name matches action

        if action == "pause":
            # Send command to Receiver process via command queue
            success = pipeline_info.send_command("pause")
        elif action == "resume":
             success = pipeline_info.send_command("resume")
        elif action == "update":
             new_settings = data.get("model_settings", {})
             if not new_settings:
                  await self.send_error(websocket, "Missing 'model_settings' in data for update action.")
                  return
             # Send command with settings data
             success = pipeline_info.send_command("update_settings", new_settings)
        else:
             await self.send_error(websocket, f"Invalid pipeline control action: {action}")
             return

        if success:
             await self.send_success(websocket, {"message": f"Action '{action}' sent to pipeline {symbol}/{model_name}"})
        else:
              await self.send_error(websocket, f"Failed to send action '{action}' to pipeline {symbol}/{model_name}. Check logs.")


    # --- WebSocket Handler ---

    async def handler(self, websocket, path):
        """Main WebSocket connection handler for each client."""
        await self._register_client(websocket)
        try:
            # Message handling loop
            async for message_str in websocket:
                # Ensure client is still considered active before processing
                if websocket not in self.client_pipelines:
                     logger.warning(f"Received message from unregistered client {websocket.remote_address}. Ignoring.")
                     break # Exit loop if client was unregistered concurrently

                await self.handle_message(websocket, message_str)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Connection closed normally: {websocket.remote_address}")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Connection closed with error: {websocket.remote_address} - {e}")
        except Exception as e:
            logger.error(f"Unexpected error in handler for {websocket.remote_address}: {e}", exc_info=True)
        finally:
            # Ensure cleanup happens regardless of how the loop exits
            await self._unregister_client(websocket)


    async def handle_message(self, websocket, message_str: str):
        """Processes incoming WebSocket messages and routes them."""
        client_id = f"{websocket.remote_address}" # Use address for logging
        # ... (message parsing remains mostly the same as previous version) ...
        try:
            request_data = json.loads(message_str)
        except json.JSONDecodeError:
            logger.error(f"[Client {client_id}] Received invalid JSON.")
            await self.send_error(websocket, "Invalid JSON format.")
            return

        msg_type = request_data.get("type")
        action = request_data.get("action")
        data = request_data.get("data", {})
        request_id = request_data.get("request_id")

        log_prefix = f"[Client {client_id}"
        if request_id: log_prefix += f", ReqID {request_id}"
        log_prefix += "]"
        logger.info(f"{log_prefix} Received request: type='{msg_type}', action='{action}'")

        if not msg_type or not action:
             await self.send_error(websocket, "Missing 'type' or 'action' field.", request_data)
             return

        # Route requests
        try:
            if msg_type == "model" and action == "list":
                # List only validated models from config
                valid_models = list(self.available_models.keys())
                await self.send_success(websocket, {"models": valid_models}, request_data)

            elif msg_type == "symbol":
                symbol_name = data.get("symbol_name")
                model_name = data.get("model_name")

                if not symbol_name or not model_name:
                    await self.send_error(websocket, "Missing 'symbol_name' or 'model_name' in data.", request_data)
                    return

                if action == "add":
                    model_settings = data.get("model_settings", {})
                    await self._start_pipeline(websocket, symbol_name, model_name, model_settings)
                elif action == "remove":
                     await self._stop_pipeline(websocket, symbol_name, model_name)
                     # Success/error messages handled within _stop_pipeline
                elif action in ["pause", "resume", "update"]:
                     action_data = data if action == "update" else None
                     await self._control_pipeline(websocket, symbol_name, model_name, action, action_data)
                     # Success/error messages handled within _control_pipeline
                else:
                    await self.send_error(websocket, f"Unknown action '{action}' for type 'symbol'.", request_data)
            else:
                 await self.send_error(websocket, f"Unknown message type '{msg_type}'.", request_data)

        except Exception as e:
             logger.error(f"{log_prefix} Unhandled error processing request (type='{msg_type}', action='{action}'): {e}", exc_info=True)
             await self.send_error(websocket, f"Internal server error processing your request.", request_data)


    async def _shutdown_service(self):
        """Graceful shutdown procedure for the main service."""
        if self.shutdown_event.is_set():
             return # Already shutting down
        logger.info("Initiating graceful shutdown...")
        self.shutdown_event.set()

        # Stop WebSocket server from accepting new connections
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            logger.info("WebSocket server stopped.")

        # Stop all client listeners and pipelines
        logger.info("Stopping all client handlers and pipelines...")
        active_clients = list(self.client_pipelines.keys())
        await asyncio.gather(*(self._unregister_client(ws) for ws in active_clients), return_exceptions=True)
        logger.info("All client resources cleaned up.")

        # Shutdown the global multiprocessing manager
        if glob_manager:
            logger.info("Shutting down global multiprocessing manager...")
            # Run synchronous shutdown in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, glob_manager.shutdown)
            logger.info("Global multiprocessing manager shut down.")

        logger.info("Shutdown complete.")


    async def run(self):
        """Starts the WebSocket server and handles graceful shutdown."""
        host = self.config.get('server', {}).get('host', '0.0.0.0')
        port = self.config.get('server', {}).get('port', 8765)

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._shutdown_service()))

        logger.info(f"Starting WebSocket server on ws://{host}:{port}")
        try:
            async with websockets.serve(self.handler, host, port) as server:
                self.websocket_server = server # Store server instance
                logger.info("Server started successfully. Waiting for connections or shutdown signal...")
                await self.shutdown_event.wait() # Wait until shutdown is triggered
        except OSError as e:
             logger.critical(f"Failed to start server on {host}:{port}: {e}. Port likely in use.")
        except Exception as e:
             logger.critical(f"Unexpected error starting or running server: {e}", exc_info=True)
        finally:
            logger.info("Server run loop exiting...")
            # Ensure shutdown runs even if server fails to start/run cleanly
            if not self.shutdown_event.is_set():
                 await self._shutdown_service()


if __name__ == "__main__":
    multiprocessing.freeze_support() # Add this for Windows compatibility

    # Determine config file path relative to this script
    script_dir = Path(__file__).parent
    default_config_path = script_dir.parent / "service_conf.yaml"
    config_file = default_config_path 

    # Basic logging setup before config load
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    service = None
    try:
        service = DataAgentService(config_path=config_file)
        asyncio.run(service.run())
    except FileNotFoundError as e:
        logger.critical(f"Service startup failed: {e}")
    except (ValueError, RuntimeError) as e: # Catch config/manager errors
        logger.critical(f"Service startup failed due to invalid configuration or setup: {e}")
    except Exception as e:
         logger.critical(f"Unhandled exception during service startup or execution: {e}", exc_info=True)
    finally:
         logger.info("Service exiting.")
         # Final check for manager shutdown if service failed very early
         if service is None and glob_manager and hasattr(glob_manager, '_process'):
             try:
                  logger.warning("Attempting manager shutdown after early service exit.")
                  glob_manager.shutdown()
             except Exception:
                  logger.exception("Error during final manager shutdown attempt.")
                  