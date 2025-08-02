import io
import os
import torch

import numpy as np
import whisper
import librosa
import ffmpeg
from opencc import OpenCC

#from server.console import log

model_root=R"E:\Program\$knowlage\$AI\Model\whisper-STT"
model_level="medium.pt"


# log("voice loading checkpoint "+model_level+" ...","voice")
model = whisper.load_model(os.path.join(model_root, model_level))
# log("loading successful","voice")

cc = OpenCC('t2s')


def bin_decode(bin_array,code_type="wav"):
    PCM_array = None
    if code_type == "wav":
        PCM_array,_ = librosa.load(
            io.BytesIO(bin_array),  # 二进制数据转文件流
            sr=16000,  # 可选：重采样到 16kHz（Whisper 推荐）
            mono=True  # 转为单声道
        )
    if code_type == "webm":
        PCM_array,_ = (
            ffmpeg
            .input('pipe:0', format='webm')  # 从 stdin 读取 WebM 数据
            .output(
                'pipe:1',  # 输出到 stdout
                format='f32le',  # 32-bit 浮点 PCM
                ac=1,  # 单声道
                ar=16000  # 16kHz 采样率
            )
            .run(input=bin_array, capture_stdout=True, quiet=True)
        )
        PCM_array = np.frombuffer(PCM_array, dtype=np.float32)

    return PCM_array
def voice2text(audio_array):
    # log("get voice to text requirement","voice")


    result = model.transcribe(audio=torch.tensor(audio_array, dtype=torch.float32))
    content = result["text"]

    # log("transcribe successful!","voice")
    return cc.convert(content)

def text2voice(text):
    # log("text2voice start","voice")
    return


audio = open("晋升后交谈1.webm", "rb").read()


print(voice2text(bin_decode(audio,"webm")))






