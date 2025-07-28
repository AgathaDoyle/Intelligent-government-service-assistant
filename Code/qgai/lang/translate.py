import json
import os

__all__=['translate','detranslate']


def translate(vocab,lang="zh-cn"):
    lang_dic = json.loads(open(os.path.join("lang",lang+".json"),'r',encoding='utf-8').read())

    if vocab not in lang_dic:
        return None
    else:
        return lang_dic[vocab]


def detranslate(vocab,lang="zh-cn"):
    lang_dic = json.loads(open(os.path.join("lang",lang+".json"),'r',encoding='utf-8').read())

    lang_dic = dict(zip(lang_dic.values(), lang_dic.keys()))

    if vocab not in lang_dic:
        return None
    else:
        return lang_dic[vocab]
