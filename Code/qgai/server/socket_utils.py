import websockets
import asyncio
import threading


from console import log
import face
import voice



async def utils_handle(websocket:websockets.ClientConnection, path=""):
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

        total = 0
        async for img in websocket:
            # 截取特征部分
            feature = face.face_fetcher(img)
            if feature is None:
                continue
            # 如果特征不为none识别特征部分
            else:
                predict_id = face.cv2_predict([img], min_acc=0.5)
                user_id_counter[predict_id] = user_id_counter[predict_id]+1 if predict_id in user_id_counter.keys() else 1
                total += 1

            # 识别次数足够
            if total > need_times:
                max_user = list(user_id_counter.keys())[0]
                count = list(user_id_counter.values())[0]
                # 计算最大准确率者
                for key in user_id_counter:
                    if user_id_counter[key]>user_id_counter[max_user]:
                        max_user = key
                        count = user_id_counter[max_user]

                # 如果低于阈值，重新开始
                if count <need_times*acc:
                    user_id_counter = {}
                # 识别成功，响应
                else:
                    user_id = max_user
                    message = "success"

            response = {
                "user_id": user_id,
                "type": "face_predict",
                "hash": "",

                "message": message
            }

            # 发送响应信息
            await websocket.send(response)
            if message == "success":
                await websocket.close(200,"success")
                return

    elif path == "/tts":
        async for text in websocket:

    elif path == "/stt":
        async for sound in websocket:
            pcm = voice.bin2pcm(sound)
            text = voice.voice2text(pcm)
            await websocket.send(text)








async def begin():
    async with websockets.serve(handler=utils_handle, host="192.168.58.1", port=33042):
        await asyncio.Future()


def run():
    threading.Thread(target=asyncio.run,args = (begin(),)).start()\









