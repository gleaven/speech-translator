import socket
from rich.console import Console
import logging

logger = logging.getLogger(__name__)

console = Console()


class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(self, stop_event, queue_in, host="0.0.0.0", port=12346):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)

        while not self.stop_event.is_set():
            logger.info("Sender waiting to be connected...")
            self.conn, _ = self.socket.accept()
            logger.info("sender connected")

            while not self.stop_event.is_set():
                audio_chunk = self.queue_in.get()
                if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                    break
                try:
                    self.conn.sendall(audio_chunk)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    logger.info("Sender client disconnected mid-send")
                    # Drain any stale audio chunks from the queue
                    while not self.queue_in.empty():
                        self.queue_in.get()
                    break
            self.conn.close()
            logger.info("Sender client disconnected, waiting for new connection...")
