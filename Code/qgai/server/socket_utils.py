import websockets
import asyncio
import threading


from console import log
import face
import voice



async def handle(websocket:websockets.ClientConnection,path=""):
    if path == "":
        await websocket.send("please add path")
        await websocket.close(400,"please add path")
        return

    elif path == "/face_predict":
        user_id_counter={}
        need_times = 50
        acc = 0.5
        user_id = ""
        message = "continue"

        async for img in websocket:
            feature = face.face_fetcher(img)
            if feature is None:
                continue
            else:
                predict_id = face.cv2_predict([img], min_acc=0.5)
                user_id_counter[predict_id] = user_id_counter[predict_id]+1 if predict_id in user_id_counter.keys() else 1


            if len(user_id_counter.keys()) > need_times:
                max_user = list(user_id_counter.keys())[0]
                count = list(user_id_counter.values())[0]
                for key in user_id_counter:
                    if user_id_counter[key]>user_id_counter[max_user]:
                        max_user = key
                        count = user_id_counter[max_user]

                if count <need_times*acc:
                    user_id_counter = {}
                else:
                    user_id = max_user
                    message = "success"

            response = {
                "user_id": user_id,
                "type": "face_predict",
                "hash": "",

                "message": message
            }

            await websocket.send(response)
            if message == "success":
                await websocket.close(200,"success")
                return








async def begin():
    async with websockets.serve(handler=handle, host="192.168.58.1", port=33042):
        await asyncio.Future()


def run():
    threading.Thread(target=asyncio.run,args = (begin(),)).start()\









