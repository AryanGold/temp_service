import asyncio
import websockets
import logging
import multiprocessing
import os

import utils
import conf
import logging_setup
from ClientHandler import client_handler
from DataProviders.ThetaData.ThetaData import ThetaData

conf_file = 'service_conf.yaml'

async def main():
    conf.glob_manager = multiprocessing.Manager()

    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    conf.data = utils.load_config(curr_dir_path + '/' + conf_file)

    # Load Data Provider
    # We load data provider here for allow warmup data provider before clients connected.
    # Specifically ThetaData require starting 'ThetaTerminal.jar' which may take about 3 seconds.
    conf.dp = ThetaData()
    conf.dp.start()

    # Main WebSocket server that spawns per-client pipelines.
    server = await websockets.serve(client_handler, "localhost", 8765)
    logging.info("[Service] WebSocket Server Started. Waiting for clients...")
    
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        pass
    finally:
        logging.info("[Service] WebSocket Server closing...")
        server.close()
        await server.wait_closed()
        logging.info("[Service] WebSocket Server close done...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("[Service] Shutting down...")

        # Terminate all spawned processes
        for client_id, pipeline in conf.processes.items(): 
            for pipeline_key, processes in pipeline.items(): 
                for p in processes:
                    p.terminate()
                    p.join()