import socket
from contextlib import closing
import carla
import os

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

free_port = find_free_port()

client = carla.Client('localhost', 2000, worker_threads=1)
client.set_timeout(10.0)

world = client.load_world('Town06')
