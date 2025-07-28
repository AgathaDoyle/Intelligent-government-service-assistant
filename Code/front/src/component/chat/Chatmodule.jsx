import React, { useState, useRef, useEffect } from "react";
import { Button, Input, message } from "antd";
import {
  SendOutlined,
  AudioOutlined,
  AudioMutedOutlined,
} from "@ant-design/icons";
import styles from "./ChatModule.module.css";
import { sendMessage, receiveMessages } from "../../api/userservice/chat.js";
// import { encrypt } from "../../component/secret/encrypt.js";
const { TextArea } = Input;
import { jsPDF } from "jspdf";
import "jspdf-autotable";
import { encryptData } from "../../utils/encrypt.js";

let key = ""//暂存key
let flow = ""//暂存flow

const ChatModule = ({ onSend, initialMessages = [], getTitle }) => {
  // const [messages, setMessages] = useState(initialMessages);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isListening, setIsListening] = useState(false);
  // const [activated, setActivated] = useState(initialMessages.length > 0);
  const [activated, setActivated] = useState(false);
  const messageEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const scrollRef = useRef(null);
  const [isSending, setIsSending] = useState(false);
  const user = JSON.parse(localStorage.getItem("user"));
  const userId = user.id; // 假设用户ID存

  // 语音识别配置
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true; // ✅ 允许持续识别直到用户手动停止
      recognition.interimResults = false; // 如果需要实时中间结果可以设为 true
      recognition.lang = "zh-CN";

      recognition.onstart = () => {
        setIsListening(true);
        message.info("开始语音识别...");
      };

      recognition.onend = () => {
        // 注意：这只会在手动停止或出错时调用
        setIsListening(false);
      };

      recognition.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript;
          }
        }
        setInput((prev) => prev + transcript);
      };

      recognition.onerror = (event) => {
        setIsListening(false);
        message.error("语音识别出错：" + event.error);
      };

      recognitionRef.current = recognition;
    }
  }, []);
  // useEffect(() => {
  //   const SpeechRecognition =
  //     window.SpeechRecognition || window.webkitSpeechRecognition;
  //   if (SpeechRecognition) {
  //     const recognition = new SpeechRecognition();
  //     recognition.continuous = false;
  //     recognition.interimResults = false;
  //     recognition.lang = "zh-CN";

  //     recognition.onstart = () => {
  //       setIsListening(true);
  //       message.info("开始语音识别...");
  //     };

  //     recognition.onend = () => {
  //       setIsListening(false);
  //     };

  //     recognition.onresult = (event) => {
  //       const transcript = event.results[0][0].transcript;
  //       setInput((prev) => prev + transcript);
  //       message.success("语音识别完成");
  //     };

  //     recognition.onerror = (event) => {
  //       setIsListening(false);
  //       message.error("语音识别失败，请重试");
  //     };

  //     recognitionRef.current = recognition;
  //   }
  // }, []);

  // 发送消息

  const handleSend = async () => {
    setIsSending(true); // 禁用发送
    if (recognitionRef.current != null) recognitionRef.current.stop(); // 停止语音识别
    if (input.trim()) {
      const newMessage = {
        content: input.trim(),
        role: "user",
        timestamp: new Date(),
      };
	  
	  setInput("");
      //如果为第一条信息，则调用getTitle函数
      if (messages.length === 0) {
        getTitle(newMessage.content);
		//发送信息
		try {
		  //! 记得加密解密
		  setActivated(true);
		  setMessages((prev) => [...prev, newMessage]);
		  const id = user.id;
		  delete user.id; // 删除id字段
		  const response = await sendMessage({
		    user_id: id,
		    info: {
				name:user.name,
				gender:user.gender,
				nationality:user.nationality,
				phone:user.phone,
				id_card:user.idCard
			},
			type: "handshake",
		    input: { type: "text", text: newMessage.content },
		  });
		  user.id = id; // 恢复id字段
		  flow = response.flow;
		  //这里注意ai返回的信息格式
		} catch (error) {
		  message.error("发送信息失败，请稍后重试");
		  setMessages((prev) => [
		    ...prev,
		    {
		      content: "发送失败，请稍后重试",
		      role: "assistant",
		      timestamp: new Date(),
		    },
		  ]);
		}
      }
	  else
	  {
		  console.log(2)
		  //用户回答
		  try {
		    //! 记得加密解密
		    setActivated(true);
		    setMessages((prev) => [...prev, 
			{
				content: input.trim(),
				role: "user",
				timestamp: new Date(),
			}]);
		    const response = await sendMessage({
		      user_id: user.id,
			  type: "answer",
			  input: {
				  type: "text",
				  text: newMessage.content,
				  key: key
			  }
		    });
			newMessage.content = response.output.value
		    setMessages((prev) => [...prev, {
				content: response.output.value,
				role: "user",
				timestamp: new Date(),
				
			}]);
		  setIsSending(false);
		  } catch (error) {
		  	console.log(error)
		    message.error("发送信息失败，请稍后重试");
		    setMessages((prev) => [
		      ...prev,
		      {
		        content: "发送失败，请稍后重试",
		        role: "assistant",
		        timestamp: new Date(),
		      },
		    ]);
		    setIsSending(false);
		  }
	  }
	  //get inquire
	  try{
		  console.log(22)
		setActivated(true);
		const response = await sendMessage({
		  user_id: userId,
		  type: "question",
		  
		});
		setIsSending(false);
		if (response.output.type == null) 
		{
		  console.log(32)
			await summeryData(user.id);
		}
		key = response.output.key;
		setMessages((prev) => [
		  ...prev,
		  { 
			content: response.output.text, 
			role: "assistant",
		    timestamp: new Date() 
		  },
		]);
		
	  }catch (error) {
		  return
	    message.error("发送信息失败，请稍后重试");
	    setMessages((prev) => [
	      ...prev,
	      {
	        content: "发送失败，请稍后重试",
	        role: "assistant",
	        timestamp: new Date(),
	      },
	    ]);
	  }
	  
	  console.log("发送信息：", newMessage.content);

      // try {
      //   await changeUserData({
      //     name: "linlinlin1",
      //   });
      //   console.log("更新用户数据成功");
      // } catch (error) {
      //   console.error("更新用户数据失败", error);
      // }

      // try {
      //   const response = await getUserInfo(userId);
      //   console.log("fasongId", userId);
      //   console.log("获取用户信息成功", response);
      //   message.success("获取用户信息成功");
      // } catch (error) {
      //   console.error("获取用户信息失败", error);
      // }

      //轮询获取回复
      // pollForAIResponse(hash);
    }
  };
  
  const summeryData = async (userId) =>{
	  var response = await sendMessage({
		"user_id":userId,
		"type":"summary",
		"hash":""
	  });
	  flow = response.output.flow
	  setMessages((prev) => [
	    ...prev,
	    {
	      content: flow,
	      role: "assistant",
	      timestamp: new Date(),
	    },
	  ]);
  console.log(111111111111111111111111111111111111)
	  generatePDF(response.output)
  };
  
  const generatePDF = (output) => {
    // 创建PDF文档
    const doc = new jsPDF();
  
  console.log(555555555555555555555555555)
    // 1. 添加标题
    doc.setFontSize(18);
    doc.text("政务服务办理摘要", 105, 20, { align: "center" });
  
    // 2. 添加分类信息
    doc.setFontSize(14);
    doc.text(`业务分类: ${output.classify}`, 15, 40);
  
  console.log(44444444444444444444)
    // 3. 添加表格数据
    doc.setFontSize(12);
    doc.text("相关信息表格:", 15, 60);
  console.log(output.tables[0])
    // 转换表格数据为AutoTable需要的格式
    const tableData = output.tables[0].map((item) => [
      item["姓名"] || "",
      item.age || "",
      item.gender || "",
      item.idNumber || "",
    ]);
  
  console.log(3333333333333333333333333333)
    // 添加表格
    doc.autoTable({
      startY: 65,
      head: [["姓名", "年龄", "性别", "身份证号"]],
      body: tableData,
      theme: "grid",
      headStyles: {
        fillColor: [22, 160, 133],
        textColor: 255,
      },
    });
  
  console.log(7777777777777777777777777777)
    // 4. 添加办理流程（自动处理换行）
    doc.setFontSize(12);
    doc.text("办理流程:", 15, doc.autoTable.previous.finalY + 20);
  
    // 处理流程文本的换行
    const splitText = doc.splitTextToSize(output.flow, 180);
    doc.text(splitText, 15, doc.autoTable.previous.finalY + 30);
  
  console.log(666666666666666666666666666666666666)
    // 5. 添加生成时间
    const date = new Date().toLocaleString();
    doc.setFontSize(10);
    doc.text(`生成时间: ${date}`, 15, doc.internal.pageSize.height - 10);
  
  console.log(2222222222222222222222222222222222)
    // 保存PDF
    doc.save(`政务服务摘要_${output.classify}.pdf`);
  };


    // const poller = setInterval(() => {
    //   retries++;

    //   // 模拟 AI 返回内容（你可以替换为实际 fetch 请求）
    //   const mockAIResponse = {
    //     content: "这是AI的回复内容。",
    //     role: "assistant",
    //     timestamp: new Date(),
    //   };

    //   // 模拟第3次轮询收到回复
    //   if (retries === 3) {
    //     clearInterval(poller);
    //     setMessages((prev) => [...prev, mockAIResponse]);
    //     setIsSending(false); // 启用发送按钮

    //     // 朗读文字
    //     speakText(mockAIResponse.content);
    //   }

    //   if (retries >= maxRetries) {
    //     clearInterval(poller);
    //     setIsSending(false);
    //     message.error("AI未及时回复，请稍后重试");
    //   }
    // }, interval);
  

  //将文字转语音
  function speakText(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "zh-CN";
    window.speechSynthesis.speak(utterance);
  }

  // 开始语音识别
  const startListening = () => {
    if (!recognitionRef.current) {
      message.error("您的浏览器不支持语音识别功能");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      try {
        recognitionRef.current.start();
      } catch (error) {
        message.error("启动语音识别失败");
      }
    }
  };

  // 滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // 处理键盘事件
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 渲染输入区域
  const renderInputArea = () => (
    <div
      className={activated ? styles.chatFooter : styles.initialInputContainer}
    >
      <div className={styles.inputArea}>
        <TextArea
          className={styles.textArea}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          autoSize={{ minRows: 2, maxRows: 5 }}
          placeholder="请输入内容..."
        />
        <div className={styles.chatActions}>
          <Button
            icon={isListening ? <AudioMutedOutlined /> : <AudioOutlined />}
            onClick={startListening}
            className={`${styles.voiceButton} ${
              isListening ? styles.listening : ""
            }`}
            type={isListening ? "primary" : "default"}
            danger={isListening}
          >
            {isListening ? "停止录音" : "语音输入"}
          </Button>
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            disabled={!input.trim() || isSending} // 添加 isSending 条件
            className={styles.sendButton}
          >
            发送
          </Button>
        </div>
      </div>
    </div>
  );

  // 如果未激活，显示初始居中布局
  if (!activated) {
    return (
      <div className={styles.chatWrapper}>
        <div className={styles.initialState}>
          <h1 className={styles.title}>欢迎来到多智能体协同辅助政务服务助手</h1>
          {renderInputArea()}
        </div>
      </div>
    );
  }

  // 激活后的布局
  return (
    <div className={styles.chatWrapper}>
      <div className={styles.activatedLayout}>
        <div className={styles.chatContent} ref={scrollRef}>
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`${styles.chatMessage} ${
                msg.role === "user" ? styles.messageRight : styles.messageLeft
              }`}
            >
              <div
                className={
                  msg.role === "user"
                    ? styles.chatMessageUser
                    : styles.chatMessageAI
                }
              >
                {msg.content}
              </div>
            </div>
          ))}
          <div ref={messageEndRef} />
        </div>
        {renderInputArea()}
      </div>
    </div>
  );
};

export default ChatModule;
