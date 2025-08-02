import http
import json
import hashlib
import base64

import http.server as httpserver


from user import User,TableFiller,AsyncModel,AsyncPredictLooper,AsyncTrainLooper,LoopBed

from server.console import log
from server.cipher import AESCipher,ChiperBase
from lang import translate
from variable_pool import error_response,processing_response


from server.variable_pool import http_host,http_port,user_dic,loop_beds

__all__=['run']

ip= http_host + ":" + str(http_port)
hash_obj = hashlib.sha256()



class Handler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    def do_GET(self):
        self.wfile.write(b"\x59\x7D\x54\xE5\x54\xE5\x4E\xBA\x5B\xB6\x60\xF3\x89\x81\x4F\x60post\x62\x11\xFF\x0C\x4E\x0D\x89\x81get\x4E\x0D\x89\x81get\x4E\x0D\x89\x81get\x4E\x0D\x89\x81get\x4E\x0D\x89\x81get\x4E\x0D\x89\x81get")
        self.wfile.flush()
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



        # 握手时创建新用户
        if user_id not in user_dic and req_type == "handshake":
            log("(%s)a new user handshake" % ip)

            user = User(request["info"],LoopBed().looping_on_new_thread())
            user_dic[user_id] = user
            user = user_dic[user_id]



        # 在握手前使用，报错
        if user_id not in user_dic:
            response["type"] = "error"
            response["message"]="use this user_in before handshake"
            return response


        # 获取用户
        user = user_dic[user_id]
        # 初始化握手响应
        if req_type == "handshake":
            log("handshake----user_id:%s" % user_id)

            user.init(request["input"]["text"])
            if not user.init.done:
                return processing_response(user_id, req_hash)
            else:
                response["type"] = "handshake"
                response["classify"] = user.bus_type
                response["flow"] = user.flow
                user.classify(request["input"]["text"])

        #数据库储存请求
        elif req_type == "storage":
            response["type"] = "storage"
            response["output"] = user.main_info
        #总结
        elif req_type == "summary":
            flow = user.get_flow()

            if not user.get_flow.done:
                return processing_response(user_id,req_hash)

            response["type"] = "summary"
            response["output"] = {
                "tables":user.tables_filler.tables,
                "classify":user.bus_type,
                "flow":flow
            }
        return response


def run():
    web = httpserver.ThreadingHTTPServer((http_host, http_port), Handler)
    log("a new http listener open on %s"%ip)
    web.serve_forever()
    while True:
        command = input("console@%s:~$"%ip)