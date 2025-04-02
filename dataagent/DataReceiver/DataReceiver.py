import multiprocessing
import asyncio
import websockets
import json
import random
import time
import logging

import conf
from . import DataPrepare
import ModelBase

# Receives data and pushes it into the emitter queue
class DataReceiver(multiprocessing.Process):
    def __init__(self, client_id, emitter_queue, pipeline_key):
        super().__init__()
        self.client_id = client_id
        self.clientdata_manager = conf.clients[client_id]
        self.emitter_queue = emitter_queue
        self.pipeline_key = pipeline_key
        self.model = None

        self.isWarmup = False

    def fetch_quotes_data(self, symbol):
        res = conf.dp.get_chain(symbol)
        if res is None:
            return None
        chain, provider_latency_ms = res

        chain_obj = DataPrepare.chain_prepare(chain, rates=0.05, expiration_time='16:00:00', calc=True)

        return chain_obj, provider_latency_ms

    def run(self):
        client_data = self.clientdata_manager.get(self.pipeline_key)
        symbol = client_data["symbol"]
        model_name = client_data["model_name"]

        self.model = ModelBase.load_model(client_data)
        if self.model is None:
            logging.error(f"[DataReceiver][{self.client_id}] undefined model: {client_data["model_name"]}")
            return

        # Wait until ThetaTerminal.jar start
        time.sleep(1)

        while True:
            res = self.fetch_quotes_data(symbol)
            if res is None:
                time.sleep(conf.data["data_receiver"]["fetch_data_delay"])
                continue

            chain_obj, provider_latency_ms = res

            if chain_obj is not None:
                model_calibrate = None

                ###
                # Get Model calibration paramters
                if not self.isWarmup:
                    # First time we calculate calibration in this process for Warmup model
                    self.isWarmup = True

                    # This calibration perfrom only during warmup, all other calls perform in DataProcessor
                    model_calibrate = self.model.calibrate_model_params_to_chain(chain_obj)
                else:
                    # When model already warm we obtaon calibration data from DataRpocessor module
                    client_data = self.clientdata_manager.get(self.pipeline_key)
                    model_calibrate = client_data["model_calibrate"]
                ###

                new_chain_obj = self.model.evaluate_model_for_chain(chain_obj, model_calibrate)

                ###
                # Store neccessary data to cache for DataProcessor module
                # DataProcessor do heavy calculations
                lock = self.clientdata_manager.acquire_lock(self.pipeline_key)
                with lock:
                    client_data = self.clientdata_manager.get(self.pipeline_key)
                    # Store calibration parameters for use it in DataProcessor
                    client_data["model_calibrate"] = model_calibrate
                    # Store chain object for use it in DataProcessor
                    client_data["new_chain_obj"] = new_chain_obj

                    self.clientdata_manager.update(self.pipeline_key, client_data)
                ###

                #new_chain_obj.chain.to_csv(f"logs/chain_obj.csv")

                result = {
                    "client_id": self.client_id,
                    "symbol": symbol,
                    "model": model_name,
                    "provider_latency_ms": provider_latency_ms
                }

                self.emitter_queue.put(result)

            time.sleep(conf.data["data_receiver"]["fetch_data_delay"])