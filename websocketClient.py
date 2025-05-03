import asyncio
import websockets
import json
import argparse

class TicTacToeClient:
    def __init__(self, uri):
        self.uri = uri
        self.websocket = None

    # Connect to the server
    async def connect(self):
        self.websocket = await websockets.connect(self.uri, ping_interval=None, ping_timeout=None)
        print(f"Connected to the server at {self.uri}")

    # Disconnect from the server
    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from the server.")

    # Receive a message from the server (can be last_move or opponent's move)
    async def receive_move_message_or_none(self):
        msg = await self.websocket.recv()
        try:
            server_msg = json.loads(msg)
            print("Received JSON from server:", server_msg)
            return server_msg
        except json.JSONDecodeError:
            print("Received non-JSON message:", msg)
            return None  # or handle this case as needed

    # Send a move to the server
    async def make_move(self, move):
        move_msg = {"move": list(move)}
        await self.websocket.send(json.dumps(move_msg))
        print("Sent to server:", move_msg)

async def main(uri):
    client = TicTacToeClient(uri)

    # Connect to the server
    await client.connect()

    while True:
        # Get initial last_move or game state
        server_msg = await client.receive_move_message_or_none()
        if(server_msg is None):
            return None  # handle "Winner X" => restart model

        last_move = server_msg.get("last_move", None)
        print("Last move from server:", last_move)

        # Example move based on last move, if it's None, use a dummy move
        my_move = (0, 0) if last_move is None else (last_move[1], last_move[0])  # dummy move

        # Make a move
        await client.make_move(my_move)

    # Disconnect from the server
    await client.disconnect()

if __name__ == "__main__":
    # Argument parsing to allow passing WebSocket URI from command line
    parser = argparse.ArgumentParser(description="Connect to the Tic-Tac-Toe WebSocket server.")
    parser.add_argument("uri", help="WebSocket server URI (e.g., ws://localhost:PORT)")

    args = parser.parse_args()

    # Run the client with the provided WebSocket URI
    asyncio.run(main(args.uri))
