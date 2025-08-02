import websockets
import json
import asyncio
import threading

from websockets import ClientConnection
from variable_pool import user_dic,socket_process_host,socket_process_port
from variable_pool import processing_response,error_response
from user import User

async def qna_poster(websocket:ClientConnection,user_id:str,hash):
    while True:
        try:
            question = yield
            await websocket.send(question)

            yield await websocket.recv(True)

            table = yield
            await websocket.send(table)
        except websockets.exceptions.ConnectionClosedError:
            await websocket.close()

async def train_poster(websocket:ClientConnection,user_id:str,hash):
    while True:
        try:
            async for img in websocket:
                yield img
                message = yield

                response = {
                    "user_id":user_id,
                    "hash":hash,
                    "type":"face_train",

                    "message":message
                }

                if message == "success":
                    await websocket.close()

        except websockets.exceptions.ConnectionClosedError:
            await websocket.close()


async def process_handler(websocket:ClientConnection, path):
    handshake = json.loads(await websocket.recv(True))
    user_id = handshake['user_id']
    hash = handshake['hash']
    type = handshake['type']

    if not user_id in user_dic:
        await websocket.send(json.dumps(error_response(user_id, hash,"please handshake before connect socket")))
        await websocket.close(400,"please handshake before connect socket")
        return

    user:User = user_dic[user_id]
    if type == "face_train":
        train_gen = train_poster(websocket,user_id,hash)
        state = await user.train_hosting(train_gen,user_id)
    if type == "qna":
        qna_gen = qna_poster(websocket,user_id,hash)
        state = await user.qna_hosting(qna_gen)

    await websocket.close(200,"finish")


async def begin():
    async with websockets.serve(process_handler, "ws://%s:%d" %(socket_process_host,socket_process_port)):
        await asyncio.Future()

def run():
    threading.Thread(target=asyncio.run,args = (begin(),)).start()