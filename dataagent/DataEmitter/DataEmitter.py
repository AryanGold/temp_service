import asyncio
import threading
import websockets
import logging

# Sends processed data via a pipe to the main WebSocket process.
class DataEmitter:
    def __init__(self):
        self.client_queues = {}
        self.client_websockets = {}

    def register_client(self, client_id, queue, websocket):
        self.client_queues[client_id] = queue
        self.client_websockets[client_id] = websocket
        asyncio.create_task(self.start_emitting(client_id))

    async def start_emitting(self, client_id):
        queue = self.client_queues[client_id]
        websocket = self.client_websockets[client_id]

        try:
            while True:
                result = await asyncio.get_event_loop().run_in_executor(None, queue.get)
                if result is None:
                    break  # Shutdown signal
                await websocket.send(str(result))
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"[Emitter][{client_id}] WebSocket closed")
        except Exception as e:
            logging.info(f"[Emitter][{client_id}] Exception in emitter: {e}")
        finally:
            logging.info(f"[Emitter][{client_id}] Emitter exiting.")
