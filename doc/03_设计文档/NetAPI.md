# 平台对接API

前端，后端，py三端通讯

## 工具模块（开放在port=3304）（无需握手的，无需提供任何信息的，快速的ai-socket接口）

### 人脸识别(path=/face_predict)

- ##### socket-send

     ```json
     直接传一张图片二进制
     ```

- ##### socket-recieve

     未检测完成，继续读取

     ```json
     {
     	"user_id":"",
         "type":"face_predict",
         "hash":"a1dadafgdas3asd4s2sfad",
         
         "message":"continue"
     }
     ```

     为获取成功

     ```json
     {
     	"user_id":"asdasd",
         "type":"face_predict",
         "hash":"a1dadafgdas3asd4s2sfad",
         
         "message":"success"
     }
     ```

     请求次数过多，进行完此次回复，服务器将会断开socket

     ```json
     {
     	"user_id":"asdasd",
         "type":"face_predict",
         "hash":"a1dadafgdas3asd4s2sfad",
         
         "message":"fail"
     }
     ```

### 语音转文字(path=/stt)

- ##### socket-send

     ```json
     直接传入音频二进制
     ```

- ##### socket-recieve

     ```json
     "音频的文本"
     ```

### 语音转文字(path=/tts)

- ##### socket-send

     ```json
     "要转换的文本"
     ```

- ##### socket-recieve

     ```json
     音频二进制
     ```

## 流程（socket开放在port=444，http在port=10925）

### 首次连接（握手）(==此握手在所有流程请求前是必要的==)

- ##### post（后端此处要查询数据库，把所有用户信息（必要+基本）放到info，没有的填None）

     文本形式

     ```json
     {
         "user_id":"asdasd",
         "type":"handshake",
     
         //必要信息+基本信息
         "info":{
             "name":"a",
             "age":18,
             "birth":20250721,
             ".....":"....."
         },
     
     
         "input":{
             "type":"text",	
             "text":"我要办身份证"
         }
     }
     ```

     语音形式

     ```json
     {
         "user_id":"asdasd",
         "type":"handshake",
     
         "info":{
             "name":"a",
             "age":18,
             "birth":20250721,
             ".....":"....."
         },
     
     
         "input":{
             "type":"audio",	
             "media":"音频二进制格式"
         }
     }
     ```

- ##### response

     ```json
     {
     	"user_id":"asdasd",
         "type":"handshake",
         "hash":"a1dadafgdas3asd4s2sfad",
         
         "classify":"身份证",
         "flow":"办理身份证呢，需要先去寻找最近的政务大厅blablabla，然后带一寸照片blblbl"
     }
     ```

### 人脸录入

- ##### socket-send（handshake）握手

     ```json
     {
         user_id:"xiaofeng",		//用户id
         hash:"",				//预留，可以不实现
         type:"face_train",
     }
     ```

- ##### socket-send

     ```json
     直接传图片二进制
     ```

- ##### socket-recieve

     未检测完成，继续读取

     ```json
     {
     	"user_id":"xiaofeng",
         "type":"face_train",
         "hash":"a1dadafgdas3asd4s2sfad",
         
         "message":"continue"
     }
     ```

     为录入成功

     ```json
     {
     	"user_id":"xiaofeng",
         "type":"face_train",
         "hash":"a1dadafgdas3asd4s2sfad",
         
         "message":"success"
     }
     ```

### 对话

- ##### socket-send（handshake）

     ```json
     {
     	"user_id":"xiaofeng",
         "type":"qna",
         "hash":"a1dadafgdas3asd4s2sfad",
     }
     ```

- ##### socket-recieve（我们提问的环节）

     ```json
     文本
     ```

     无内容，即为提问结束

     ```json
     
     ```

- ##### socket-send（用户回答的环节）

     ```
     文本
     ```

### 录入（询问完成时）（总结必要信息+基本信息，将output录入数据库）

- ##### post

     ```json
     {
         "user_id":"asdasd",
         "type":"storage",
         "hash":"a1dadafgdas3asd4s2sfad"
     }
     ```

- ##### response

     ```json
     {
         "user_id":"asdasd",
         "type":"storage",
         "hash":"a1dadafgdas3asd4s2sfad",
         
     
         "output":{
             "name":"xfgg",
             "age":1234,
             "gender":"men",
             ".....":"....."
         }
     }
     ```

### 总结呈现（仅前端）

- ##### post

     ```json
     {
         "user_id":"asdasd",
         "type":"summary",
         "hash":"a1dadafgdas3asd4s2sfad"
     }
     ```

- ##### response

     ```json
     {
         "user_id":"asdasd",
         "type":"summary",
         "hash":"a1dadafgdas3asd4s2sfad",
         
     
         "output":{
             //对于这个tables，写成header和row你们更方便吗？如果方便我就改
             "tables":[
                 {
                     "name":"xfgg",
                     "age":1234,
                     "gender":"men",
                     ".....":"....."
                 },
                 {},
                 {},
                 "......"
              ],
             
             "classify":"身份证/户口本",
         	"flow":"办理身份证呢，需要先去寻找最近的政务大厅blablabla，然后带一寸照片blblbl"
         }
         
         
     }
     ```

### ==特别的响应！！！！==（仅http）

当请求正在处理中，请过几秒再请求

- ##### response

     ```json
     {
         "user_id":"asdasd",
         "type":"processing",
         "hash":"a1dadafgdas3asd4s2sfad",
     }
     ```

服务器繁忙

- ##### response

     ```json
     {
         "user_id":"asdasd",
         "type":"busy",
         "hash":"a1dadafgdas3asd4s2sfad",
     }
     ```


数据有误/用户尚未握手/流程有误

- ##### response

     ```json
     {
         "user_id":"asdasd",
         "type":"error",
         "hash":"a1dadafgdas3asd4s2sfad",
     }
     ```




# 数据库

- #### 必要信息

     ```json
     {
         "name":"英锐gg",
         "gender":"男",
         "nationality":"汉",
         "phone":"12345678901",
         "id_card":"440209200601018080"
     }
     ```

- #### 基本信息(初始留空)

     ```json
     {
         "native_place":"广州",
         "birth":"2025年8月13日",
         "email":"10925508@qq.com",
         "career":"炼丹师",
         "address":"广东省广州市番禺区广东工业大学西二-803",
         "hukou":"广州市天河区",
         "political_status":"党员",
         "marital_status":true,
         "religion":"伊斯兰教/基督教/无",
         "education":"本科"
     }
     ```
