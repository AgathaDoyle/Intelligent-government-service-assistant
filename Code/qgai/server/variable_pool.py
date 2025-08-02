from user import User,TableFiller,AsyncModel,AsyncPredictLooper,AsyncTrainLooper,LoopBed
from server.cipher import AESCipher,ChiperBase
from console import log



http_host= "192.168.58.1"
http_port=10925

socket_utils_host="192.168.58.1"
socket_utils_port=3304

socket_process_host="192.168.58.1"
socket_process_port=444


user_dic = {}
loop_beds={}

#aes_cipher = AESCipher("1234567890123456")
aes_cipher = ChiperBase()

# predict_process = AsyncPredictLooper()
# predict_process.run_loop_on_new_thread()
#
# train_process = AsyncTrainLooper()
# train_process.run_loop_on_new_thread()


def processing_response(user_id, flow_hash,message = ""):
    log("\"%s\" is request but processing" % (user_id))
    response = {
        "user_id": user_id,
        "type": "processing",
        "hash": flow_hash,

        "message": message
    }
    return response



def error_response(user_id, flow_hash,message = ""):
    log("\"%s\" is request but error" % (user_id))
    response = {
        "user_id": user_id,
        "type": "error",
        "hash": flow_hash,

        "message": message
    }
    return response

