import time
import multiprocessing
import logging

import conf
import ModelBase

# Processes data for a specific client symbol/model pair
class DataProcessor(multiprocessing.Process):
    def __init__(self, client_id, pipeline_key):
        super().__init__()
        self.client_id = client_id
        self.clientdata_manager = conf.clients[client_id]
        self.pipeline_key = pipeline_key

    def run(self):
        client_data = self.clientdata_manager.get(self.pipeline_key)
        logging.info(f"[DataProcessor][{self.client_id}] Initialize for Symbol[{client_data["symbol"]}] Model[{client_data["model_name"]}]")

        self.model = ModelBase.load_model(client_data)
        if self.model is None:
            logging.warning(f"[DataProcessor][{self.client_id}] model no loaded")
            return

        while True:
            ###
            # Load cached data
            client_data = self.load_cache()

            # Get cached calibration parameters
            model_calibrate = client_data["model_calibrate"]
            # Get cached chain object
            new_chain_obj = client_data["new_chain_obj"]
            ###

            model_calibrate_new = self.model.calibrate_model_params_to_chain(new_chain_obj, initial_guess=model_calibrate)

            ###
            # Store new calculated data to cache
            lock = self.clientdata_manager.acquire_lock(self.pipeline_key)
            with lock:
                client_data = self.clientdata_manager.get(self.pipeline_key)
                # Store calibration parameters
                client_data["model_calibrate"] = model_calibrate_new

                self.clientdata_manager.update(self.pipeline_key, client_data)
            ###

            # Small pause for models calibration which calculate very fast
            # For avoid hang system process
            time.sleep(1)

    # This method check wether model is warmup-ed.
    # When pipeline start we perform model warmup (in DataReceiver).
    # So DataProcesor wait model warmup and then begin calculation
    def load_cache(self):
        while True:
            client_data = self.clientdata_manager.get(self.pipeline_key)

            if "model_calibrate" in client_data:
                return client_data

            time.sleep(2)


