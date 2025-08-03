import functools
import threading
import json
import asyncio
import time
from typing import Optional, AsyncGenerator
import functools

from sympy import false

import inquiry
import datamining
import classify
import face
from lang import translate,detranslate

__all__=['User','TableFiller', 'AsyncModel', 'AsyncPredictLooper','AsyncTrainLooper','LoopBed']


class TableFiller:
    def __init__(self, tables):
        #表组
        self.__tables = tables

        #表格遍历迭代器的序号-1
        self.__index = -1

        #当前填写表格的指针序号
        self.__pointer = 0

    def __setitem__(self,key,value):
        """
        填写当前指针指向的表格
        """
        self.__tables[self.__pointer][key]=value


    @property
    def table(self):
        """
        返回当前所填表
        :return:
        """
        return self.__tables[self.__pointer]

    @table.setter
    def table(self, table):
        """
        修改当前所填表
        :param table: 修改内容
        :return:
        """
        self.__tables[self.__pointer] = table

    @property
    def tables(self):
        return self.__tables

    @property
    def pointer(self):
        return self.__pointer

    @property
    def is_finish(self):
        return self.__pointer>=len(self.__tables)

    @property
    def bussiness_type(self):
        return classify.type_dic

    def __iter__(self):
        self._index=-1
        return self
    def __next__(self):
        self._index+=1

        if self._index<len(self.__tables):
            return self.__tables[self._index]
        else:
            raise StopIteration
    def __getitem__(self, index):
        return self.__tables[index]

    def reset_pointer(self):
        self.__pointer=0

    def next_table(self):
        """
        导出当前所填表，并将pointer换到下一张表
        :return:
        """
        table = self.__tables[self.__pointer]
        self.__pointer+=1

        return table


class AsyncModel:
    """
    在其他线程的异步操作上分支出一个新的异步的操作，并在执行时加入循环
    """
    processing = -2147483648
    def __init__(self, task,loop:asyncio.AbstractEventLoop,wait_time=5):
        """
        创建一个异步操作模块
        :param task: 一个函数，必须为async的异步函数
        """
        self.__result = None

        self.__task = task
        self.__loop = loop

        self.__history=[]

        self.wait_time = wait_time
    def __call__(self,*args):
        return self.activate(*args)
    def __getitem__(self, index):
        """
        获取历史记录，若未执行过，则返回None
        """
        if abs(index)>len(self.__history) or index==len(self.__history):
            return None

        return self.__history[index]

    @property
    def __get_task(self):
        return self.__task

    def activate(self,*args):
        """
        调用异步函数，在异步结束前会返回processing=-2147483648
        :param args: 原函数的参数
        :return: 原函数的返回值或者processing=-2147483648
        """
        print("async activate")
        time.sleep(0.01)
        #无异步操作，则创建
        if self.__result is None:
            self.__result = asyncio.run_coroutine_threadsafe(self.__task(*args), self.__loop)
        print(self.__result.running())
        #异步操作已完成
        if self.__result.done():
            value = self.__result.result()
            print(value)
            self.__history.append(value)
            # 重置异步器
            self.__result = None
            return value
        #异步进行中
        else:
            if False and not self.__result.running():
                print("because an unexpected problem,async task isn't running,had run again now")
                self.__result = asyncio.run_coroutine_threadsafe(self.__task(*args), self.__loop)
                return self.activate(*args)
            return -2147483648

    async def async_activate(self,*args):
        value = await self.__task(*args)
        self.__history.append(value)
        return value

    @property
    def done(self):
        if self.__result is None:
            return True
        return self.__result.done()


class LoopBed:
    def __init__(self):
        self.__loop = asyncio.new_event_loop()

    @property
    def is_running(self):
        return self.__loop.is_running()

    @property
    def loop(self):
        return self.__loop

    def create_new_task(self,async_func)->AsyncModel:
        return AsyncModel(async_func, self.__loop)

    def __looping(self):
        self.__loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.__loop)
        self.__loop.run_forever()

    def looping_on_new_thread(self):
        threading.Thread(target=self.__looping).start()
        threading.excepthook = LoopBed.expection_hook
        return self

    @staticmethod
    def nonblock(wait_time=5):
        """
        用于在此床上创建非阻塞方法的装饰器，调用此函数时将会返回一个非阻塞异步模块
        :param wait_time:最大等待时间
        """
        return lambda func: AsyncModel(task=func, loop=asyncio.get_event_loop(), wait_time=wait_time)

    @staticmethod
    def expection_hook(args):
        print(f"子线程异常: {args.exc_value}")

class AsyncLooper:
    def __init__(self,func):
        self.__loop = None
        self.__async_models={}
        self.__task=func

    def __getitem__(self, id)->AsyncModel:
        return self.__async_models[id]

    def __contains__(self,id)->bool:
        return (id in self.__async_models)

    def __delitem__(self, id):
        del self.__async_models[id]


    def append(self,id):
        self.__async_models[id] = AsyncModel(self.__task, self.__loop)

    def run_loop(self):
        self.__loop = asyncio.new_event_loop()
        self.__loop.run_forever()

    def run_loop_on_new_thread(self):
        threading.Thread(target=self.run_loop).start()


class AsyncPredictLooper(AsyncLooper):
    def __init__(self):
        super().__init__(AsyncPredictLooper.async_predict_face)

    @staticmethod
    async def async_predict_face(imgs_bin)->Optional[str]:
        return face.cv2_predict(imgs_bin)

class AsyncTrainLooper(AsyncLooper):
    def __init__(self):
        super().__init__(AsyncTrainLooper.async_train_face)

    @staticmethod
    async def async_train_face(user_id,imgs_bin) -> Optional[str]:
        return face.cv2_train(imgs_bin,user_id)

default_nes_info=json.loads(open('user_info.json','r').read())['necessary']
default_adt_info=json.loads(open('user_info.json','r').read())['addition']


default_main_info=default_nes_info.copy()
default_main_info.update(default_adt_info)

class User:

    def __init__(self, info_dic: dict,loop_bed:LoopBed):
        """
        初始化一个用户
        :param info_dic: 数据库中保存的所有的信息，包含所有必要信息和已有的基本信息
        """

        #业务数据库
        self._data = datamining.DataMiningAgent()

        #直接录入必要+基本信息
        self.__main_info=info_dic

        #一次性信息
        self.__once_info_dic={}



        #当前业务流程标签
        self.__label=None

        #业务流程
        self.__flow=None

        #表组迭代器
        self.__tables_filler = None




        self.__loop_bed=loop_bed
        self.__get_answer_res= self.__loop_bed.create_new_task(self.__get_answer)
        self.__inquire_res = self.__loop_bed.create_new_task(self.__inquire)
        self.__classify_res=self.__loop_bed.create_new_task(self.__classify)
        self.__get_flow_res=self.__loop_bed.create_new_task(self.__get_flow)
        self.__train_face_res=self.__loop_bed.create_new_task(self.__train_face)
        self.__predict_face_res=self.__loop_bed.create_new_task(self.__predict_face)

        self.__init_res = self.__loop_bed.create_new_task(self.__init)


        #检查必要信息
        for key in default_nes_info:
            if key not in info_dic:
                a=1
                # raise Exception("Missing necessary information")

        #检查多于信息
        return

    def __setitem__(self, key, value):
        """
        录入用户信息
        :param key: 用户信息键
        :param value: 用户信息值
        :return:
        """
        if key in default_main_info:
            self.__main_info[key]=value
        else:
            self.__once_info_dic[key]=value



    @property
    def info(self)->dict:
        """
        用户的所有信息
        :return:
        """
        once_tb = self.once_info.copy()
        once_tb.update(self.main_info)
        res = {}
        for key in once_tb:
            zhcn = translate(key)
            res[zhcn if zhcn is not None else key]=once_tb[key]
        return res
    @property
    def main_info(self):
        """
        用户的主要信息，即必要信息+基本信息
        :return:
        """
        return self.__main_info
    @property
    def once_info(self):
        """
        用户的一次性信息，不会录入数据库，下次需要重新询问
        :return:
        """
        return self.__once_info_dic


    @property
    def label(self):
        if not self.__label is None:
            return self.__label
        if not self.classify[-1] is None:
            return self.classify[-1]

        raise Exception("please activate classify at least once first")
    @property
    def bus_type(self):
        """
        用户要办理的业务类型，将会把label对应到相应的文本
        :return:
        """
        invert_dic = dict(zip(classify.type_dic.values(), classify.type_dic.keys()))
        return invert_dic[self.label]



    @property
    def flow(self)->str:
        """
        此用户的业务流程
        :return: 用户流程的纯文本
        """
        if not self.__flow is None:
            return self.__flow
        if not self.get_flow[-1] is None:
            return self.get_flow[-1]
        raise Exception("please activate get_flow at least once first")


    @property
    def tables(self):
        """
        返回所有的表格
        :return:
        """
        return self.tables_filler.tables
    @tables.setter
    def tables(self,value:list):
        self.__tables_filler = TableFiller(value)
    @property
    def tables_filler(self)->TableFiller:
        """
        一个TableFiller类型的变量，用于填充表格
        :return: 填表迭代器
        """
        if self.__tables_filler is None:
            raise Exception("please activate get_flow at least once first")
        return self.__tables_filler
    def fill_in_table(self):
        """
        此方法将会把现在用户已有的所有，表格需要的信息填入表中，将会对传入的参数做出改变
        :param table: 所填表格，需要填写的值为None
        :return: 返回一个填完的表格
        """
        #遍历表格中的键
        for key in self.tables_filler.table:

            #遍历特殊标记
            for mark in datamining.table_mark:
                if mark is None:
                    break
            #跳过已填的值
            if not self.tables_filler.table[key] is None and not self.tables_filler.table[key] == '':
                continue

            #如果有对应的键值，填充表格
            if key in self.info:
                self.tables_filler[key]=self.info[key]
        return
    def export_tables(self):
        """
        导出所有表格为前端接口的形式
        """
        tables = []
        for table in self.tables:
            header = {}
            row = []
            for key in table:
                header[key] = key
                row.append(table[key])
            tables.append({'header':header, 'row':row})

        return tables



    async def __inquire(self):
        """
        对当前表进行询问，inquire的异步封装
        :return:
        """
        print("inquire")
        if self.tables_filler.is_finish:
            return None
        question = inquiry.inquire(self.tables_filler.table)
        if question is None:
            return None
        return list(question.keys())[0],list(question.values())[0]         #key,sentence
    @property
    def inquire(self)->AsyncModel:
        """
        询问模块的异步模块封装
        :return:
        """
        return self.__inquire_res
    def inquire_func(self):
        """
        对当前table_filler指向的表进行询问，本质为异步操作
        :return:
        """
        return self.__inquire_res.activate()


    async def __get_answer(self,text:str,key:str)->str:
        """
        对用户自然语言回答进行关键字提取，get_answer的异步封装
        :param text: 用户自然语言
        :param key: 问题键
        :return: 提取的关键字
        """
        print("get_answer")
        return inquiry.get_answer(text, key)
    @property
    def get_answer(self)->AsyncModel:
        """
        提取回答模块的异步模块封装
        :return:
        """
        return self.__get_answer_res
    def get_answer_func(self,text:str,key:str):
        """
        对用户自然语言回答进行关键字提取，本质为异步操作
        :param text: 用户自然语言
        :param key: 问题键
        :return: 结束时返回用户回答的关键字，过程中详见UserAsyncModel
        """
        return self.__get_answer_res.activate(text,key)


    async def __classify(self,require)->int:
        """
        用户业务分类的异步封装
        :param require: 用户需求的自然语言
        :return: 返回用户业务类别对应的label值
        """
        print("classify")
        self.__label = classify.classify(require, classify.type_dic)
        return self.__label
    @property
    def classify(self)->AsyncModel:
        """
        用户业务分类的异步模块封装
        :return:
        """
        return self.__classify_res
    def classify_func(self,require)->int:
        """
        用户业务分类，本质为异步操作
        :param require: 用户需求的自然语言
        :return: 结束时返回用户业务类别对应的label值，过程中详见UserAsyncModel
        """
        return self.__classify_res.activate(require)

    async def __get_flow(self)->str:
        """
        获取流程的异步封装
        :return:
        """
        print("get_flow")
        self.__flow = 1
        print("get_flow111111")
        self.__flow = self._data.get_flow(self.label, self.info)
        print("get_flow222222222222")
        return self.__flow
    @property
    def get_flow(self)->AsyncModel:
        """
        获取流程模块的异步模块封装
        :return:
        """
        return self.__get_flow_res
    def get_flow_func(self)->str:
        """
        获取用户当前分类的流程，本质为异步操作
        :return:
        """
        return self.__get_flow_res.activate(self.__label, self.info)

    async def __train_face(self, user_id, imgs_bin)->bool:
        return face.cv2_train(imgs_bin,user_id)
    @property
    def train_face(self)->AsyncModel:
        return self.__train_face_res
    def enter_face_func(self,user_id,imgs_bin)->bool:
        return self.__train_face_res.activate(user_id, imgs_bin)

    async def __predict_face(self,imgs_bin)->Optional[str]:
        return face.cv2_predict(imgs_bin)
    @property
    def predict_face(self)->AsyncModel:
        return self.__predict_face_res
    def predict_face_func(self,imgs_bin)->Optional[str]:
        return self.__predict_face_res.activate(imgs_bin)


    async def qna_hosting(self,hoster:AsyncGenerator):
        """
        询问回答环节的托管器，严格遵守NetAPI中的阻塞模式
        :param hoster:websocket的生成器对象
        :return:
        """
        while True:
            try:
                # 已完成所有表格
                if self.tables_filler.is_finish:
                    await hoster.asend("")                          # send
                    return True
                else:
                    #获取问题
                    inquire = await self.__inquire()
                    #完成此表格，下一张
                    if inquire is None:
                        self.tables_filler.next_table()
                        continue
                    else:
                        #发送问题
                        key,question = inquire
                        await hoster.asend(question)                # send

                        #获取回答
                        answer = await hoster.__anext__()           # recieve
                        value = await self.__get_answer(answer,key)
                        #填写答案
                        self[key] = value

            except StopIteration:
                return False

    async def train_hosting(self,hoster:AsyncGenerator,user_id,need_num = 50):
        """
        人脸录入环节的托管器，严格遵守NetAPI中的阻塞模式
        :param hoster:websocket
        """
        fetchers = []
        #接收图片
        async for img in hoster:                                    # recieve
            #截获数量足够
            if len(fetchers)>50:
                await self.__train_face(user_id,fetchers)
                await hoster.asend("success")                       # send
                return True
            else:
                #提取特征部分
                fetcher = face.face_fetcher(img)
                #检测到人脸，加入特征集
                if fetcher is not None:
                    fetchers.append(fetcher)
                await hoster.asend("continue")                      # send









    async def __init(self,require):
        """
        将会初始化如下内容
        classify：label
        get_flow：flow
        table.setter：tablefiller
        """
        print("init")
        self.tables = self._data.get_tables(await self.__classify(require))
        print("classify done")
        print(await self.__get_flow())

        print("flow done")
        print("init done")
        return self.label, self.flow
    @property
    def init(self)->AsyncModel:
        return self.__init_res

