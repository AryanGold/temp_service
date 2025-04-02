# Contain fucntions for income clients

import websockets
import logging
import multiprocessing
import json

import conf
from DataReceiver.DataReceiver import DataReceiver
from DataProcessor.DataProcessor import DataProcessor
from DataEmitter.DataEmitter import DataEmitter
from ClientDataManager import ClientDataManager

def init_client(client_id, init_data):
    if "symbol" not in init_data:
        logging.error(f"Error init_client()[{client_id}]: not found required 'symbol' row")
        return False

    symbols = []
    models = []

    # Parse symbol, can be as String or List of strings
    root = init_data["symbol"]
    if isinstance(root, str):
        symbols.append(root)
    elif isinstance(root, list):
        for s in root:
            symbols.append(s)
    else:
        logging.error(f"Error init_client()[{client_id}]: 'symbol' row should have type String or List of strings.")
        return False

    # Get models for evaluation
    if "model" in init_data:
        for m in init_data["model"]:
            models.append(m)
    else:
        # default model
        models.append(conf.default_model)

    clientdata_manager = ClientDataManager(conf.glob_manager)

    for symbol in symbols:
        for model in models:
            pipeline_key = symbol + '_' + model; # "AAPL_SSVI"

            # Actual data for each symbol/model pair.
            data = {
                "symbol": symbol,
                "model_name": model,
                "model": None, # Todo: initialize model here
                "conf": {}
            }

            clientdata_manager.init(pipeline_key, data)

    # Store to global clients container
    conf.clients[client_id] = clientdata_manager

    conf.processes[client_id] = {}

    return True

# Create processing pipeline for one symbol.model pair
def init_symbol_model(client_id, emitter_queue):
    clientdata_manager = conf.clients[client_id]

    for pipeline_key, data in clientdata_manager.items():
        p_recv = DataReceiver(client_id, emitter_queue, pipeline_key)
        p_proc = DataProcessor(client_id, pipeline_key)

        p_recv.start()
        p_proc.start()

        processes = [p_recv, p_proc]
        # Store prcosses to container for able to terminate it
        conf.processes[client_id][pipeline_key] = processes

# Handles new client connections and manage their processing pipeline.
async def client_handler(websocket):
    conf.client_id.inc()

    client_id = str(conf.client_id()) 

    logging.info(f"New client connected client_id[{client_id}]")

    ###
    # Read client initial parameters
    try:
        init_data = await websocket.recv()
        init_data = json.loads(init_data)

        if not init_client(client_id, init_data):
            logging.info(f"[Client][{client_id}] not inited")
            return

        logging.info(f"[Client][{client_id}] init parameters: {init_data}")

    except Exception as e:
        logging.error(f"Error[{client_id}] read client initial parameters: {e}")
        exit(-1)
    ###

    # Start emitter
    emitter = DataEmitter()
    emitter_queue = multiprocessing.Queue()
    emitter.register_client(client_id, emitter_queue, websocket)

    init_symbol_model(client_id, emitter_queue)

    try:
        await websocket.wait_closed()  # Keep connection open
    except websockets.exceptions.ConnectionClosed:
        logging.warning(f"Client[{client_id}] disconnected.")
    finally:
        logging.info(f"Client[{client_id}] cleanup...")

        emitter_queue.put(None) # Unblock emitter

        # Terminate processes for current client
        pipeline = conf.processes[client_id]
        for pipeline_key, processes in pipeline.items(): 
            for p in processes:
                p.terminate()
                p.join()
        conf.processes[client_id] = {}