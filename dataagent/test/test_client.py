import asyncio
import websockets
import json

async def test_client():
    # Connect to the WebSocket server and receive messages.
    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as ws:
            print("Connected to WebSocket server. Press Ctrl+C to exit.")

            ###
            # Send client initial parameters
            req = {
                "symbol": "AAPL"
            }

            try:
                await ws.send(json.dumps(req))
            except Exception as e:
                print(f"[Client] Init WS connection error: {e}")
                exit(-1)
            ###

            while True:
                try:
                    msg = await ws.recv()
                    print(f"\nReceived: {msg}")
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by the server.")
                    break
    except KeyboardInterrupt:
        print("\n[Client] KeyboardInterrupt received. Closing connection...")
    except Exception as e:
        print(f"[Client] Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_client())
    except KeyboardInterrupt:
        print("\n[Client] Exiting...")
