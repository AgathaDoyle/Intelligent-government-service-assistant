from server import *


if __name__ == '__main__':
    log("Welcome to the python API of AI for government assistants", __name__)
    http.run()
    socket_process.run()
    socket_utils.run()