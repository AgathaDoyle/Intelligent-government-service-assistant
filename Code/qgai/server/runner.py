import http
import json
import hashlib
import base64

import http.server as httpserver


from user import User,TableFiller,UserAsyncModel,AsyncPredictLooper,AsyncTrainLooper

from server.console import log
from server.cipher import AESCipher,ChiperBase
from lang import translate

__all__=['run']


host="192.168.1.230"
port=10925
ip=host+":"+str(port)

flow_dic = {}
hash_obj = hashlib.sha256()
#aes_cipher = AESCipher("1234567890123456")
aes_cipher = ChiperBase()

predict_process = AsyncPredictLooper()
predict_process.run_loop_on_new_thread()

train_process = AsyncTrainLooper()
train_process.run_loop_on_new_thread()



class Handler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    def do_GET(self):
        return
    def do_POST(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        # 请求内容长度
        content_length = int(self.headers.get("Content-Length", 0))

        # 请求内容
        request = json.loads(aes_cipher.base64_de_str(self.rfile.read(content_length).decode("utf8")))
        self.wfile.write(aes_cipher.str_en_base64(json.dumps(self.respond(request),ensure_ascii=False)).encode("utf8"))
        self.wfile.flush()

    def processing_response(self,user_id,flow_hash):
        log("\"%s\" is request but processing"%(user_id))
        response = {
            "user_id": user_id,
            "type": "processing",
            "hash": flow_hash,
        }
        return response
    def error_response(self,user_id,flow_hash):
        log("\"%s\" is request but error"%(user_id))
        response = {
            "user_id": user_id,
            "type": "error",
            "hash": flow_hash,
        }
        return response

    def operate_face(self,request,response)->dict:
        req_hash = request["hash"]
        user_id = request["user_id"]

        if request["input"]["type"] == "predict":
            imgs = [base64.b64decode(img.split(",")[1]) for img in request["input"]["imgs"]]
            file = open("test.jpg","wb")
            file.write(imgs[0])
            file.close()
            if not req_hash in predict_process:
                predict_process.append(req_hash)

            user_id = predict_process[req_hash](imgs)
            if not predict_process[req_hash].done:
                return self.processing_response(user_id,req_hash)

            response['user_id'] = user_id
            response["type"] = "face"
            response["output"] = {
                "type": request["input"]["type"],
                "user_id": user_id
            }

            del predict_process[req_hash]
            return response
        # 录入人脸
        elif request["input"]["type"] == "train":
            imgs = [base64.b64decode(img.split(",")[1]) for img in request["input"]["imgs"]]

            for index,img in enumerate(imgs):
                file = open("testpng/test%s.jpg"%str(index), "wb")
                file.write(img)
                file.close()

            train_process.append(user_id)
            train_success = train_process[user_id](user_id,imgs)
            if not train_process[user_id].done:
                return self.processing_response(user_id, req_hash)
            response["type"] = "face"
            response["output"] = {
                "type": "train",
                "success": train_success
            }
            return response

    def respond(self,request)->dict:
        # 用户id
        user_id = request["user_id"]
        # 请求类型
        req_type = request["type"]
        #哈希
        req_hash = hashlib.sha256(json.dumps(request).encode("utf8")).hexdigest() if "hash" not in request or request["hash"] == "" else request["hash"]

        # 响应必要信息
        response = {
            "user_id": user_id,
            "type": "busy",
            "hash": req_hash
        }


        #预测用户人脸
        if req_type == "face":
            return self.operate_face(request,response)

        # 握手时创建新用户
        if user_id not in flow_dic and req_type == "handshake":
            log("(%s)a new user handshake" % ip)

            user = User(request["info"])
            flow_dic[user_id] = user

            user = flow_dic[user_id]
            user.run_on_new_thread(request["input"]["text"])





        # 在握手前使用，报错
        if user_id not in flow_dic:
            response["type"] = "error"
            response["message"]="use this user_in before handshake"
            return response


        # 获取用户
        user = flow_dic[user_id]
        #用户正在初始化（处理信息）
        if not user.init_finish:
            return self.processing_response(user_id,req_hash)
        # 初始化握手响应
        if req_type == "handshake":
            log("handshake----user_id:%s" % user_id)

            response["type"] = "handshake"
            response["classify"] = user.bus_type
            response["flow"] = user.flow

        #询问
        elif req_type == "question":
            log("question----user_id:%s" % user_id)
            response["type"] = "question"

            question = None
            #查找空表
            while True:# 表是否填完
                if user.tables_filler.is_finish:
                    response["output"] = {"type": None}
                    break


                # 获取问题
                question = user.inquire()
                # 异步是否完成
                if not user.inquire.done:
                    return self.processing_response(user_id, req_hash)
                # 判断这张表的信息
                elif question is None:
                    # 此表填完了，换下一张表
                    user.tables_filler.export_table()
                    #填写一次已有信息
                    user.set_waiter()
                else:
                    response["type"] = "question"

                    # 响应问题-文本
                    key, sentence = question
                    response["output"] = {
                        "type": "text",
                        "key": key,
                        "text": sentence
                    }
                    break



        #解析回答
        elif req_type == "answer":
            log("answer----user_id:%s" % user_id)
            #解析请求体
            key, text, value = "", "", ""
            if request["input"]["type"] == "text":
                key = request["input"]["key"]
                text = request["input"]["text"]
            elif request["input"]["type"] == "audio":
                1 == 1
                #key = request["input"]["key"]
                #text = voice.voice2text(request["input"]["media"])


            #ai分析用户语言，提取关键字
            value = user.get_answer(text,key)
            #异步是否完成
            if not user.get_answer.done:
                return self.processing_response(user_id,req_hash)
            print(value)
            response["type"] = "answer"
            response["output"] = {
                "type": "text",
                "key": key,
                "text": text,
                "value": value
            }

            #信息已更新，录入
            user[key] = value
            user.set_waiter()
        #数据库储存请求
        elif req_type == "storage":
            response["type"] = "storage"
            response["output"] = user.main_info
        #总结
        elif req_type == "summary":
            flow = user.get_flow()

            if not user.get_flow.done:
                return self.processing_response(user_id,req_hash)

            response["type"] = "summary"
            response["output"] = {
                "tables":user.tables_filler.tables,
                "classify":user.bus_type,
                "flow":flow
            }
        return response


def run():
    web = httpserver.ThreadingHTTPServer((host, port), Handler)
    log("a new http listener open on %s"%ip)
    web.serve_forever()
    while True:
        command = input("console@%s:~$"%ip)