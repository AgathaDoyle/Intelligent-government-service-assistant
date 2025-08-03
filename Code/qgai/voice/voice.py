import io
import os
import torch

import numpy as np
import whisper
import librosa
import ffmpeg
from opencc import OpenCC
import pyttsx3
import io

#from server.console import log

model_root=R"E:\Program\$knowlage\$AI\Model"
sst_model="whisper-STT\medium.pt"
tts = pyttsx3.init()
tts.setProperty('rate', 200)
tts_voices = tts.getProperty('voices')

# log("voice loading checkpoint "+model_level+" ...","voice")
model = whisper.load_model(os.path.join(model_root, sst_model))
# log("loading successful","voice")

cc = OpenCC('t2s')


def bin2pcm(bin_array,code_type="webm"):
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
def pcm2bin(pcm_array,des_type="webm"):
    res_bin = None
    if des_type == "webm":
        res_bin,_ = (
            ffmpeg.input('pipe:0', format='f32le')
            .output(
                'pipe:1',
                format='webm'
            ).run(input=pcm_array, capture_stdout=True, quiet=True)
        )
    return res_bin

def wav2webm(wav_bin):
    res_bin = None
    res_bin, _ = (
        ffmpeg.input('pipe:0', format='wav')
        .output(
            'pipe:1',
            format='webm'
        ).run(input=wav_bin, capture_stdout=True, quiet=True)
    )
    return res_bin
def voice2text(webm_bin):
    """
    输入webm二进制格式，输出文本
    """
    # log("get voice to text requirement","voice")
    pcm_array = bin2pcm(webm_bin,"webm")

    result = model.transcribe(audio=torch.tensor(pcm_array, dtype=torch.float32))
    content = result["text"]

    # log("transcribe successful!","voice")
    return cc.convert(content)

def text2voice(text:str):
    """
    输入文本，输出webm二进制格式
    """
    tts.save_to_file(text, "%s.wav" % text)
    tts.runAndWait()

    file = open("%s.wav" % text, "rb")
    webm = wav2webm(file.read())

    return webm


# audio = open("晋升后交谈1.webm", "rb").read()
audio = text2voice("你好呀，今天天气真不错")
file = open("test1.webm","wb")
file.write(audio)
file.close()
text = voice2text(audio)

print(text)


# print(text)






