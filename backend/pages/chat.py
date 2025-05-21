import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import requests

st.set_page_config(page_title="ChatLaw")
st.title("ChatLaw")

# 预定义的模型列表（名称: 路径）
MODEL_OPTIONS = {
    "ChatLaw-法律咨询": "",
    "ChatLaw-司法摘要": "D:\\models\\Qwen2.5-0.5B-sfzy"
}


@st.cache_resource
def init_model(model_path):
    """根据选择的模型路径加载模型"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype="auto"
        )
        if torch.cuda.is_available():
            model.to('cuda')

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def response_generator(response):
    for word in response:
        yield word
        time.sleep(0.02)


# 获取Access Token的函数
def get_access_token():
    url = "https://chatglm.cn/chatglm/assistant-api/v1/get_token"
    data = {
        "api_key": "xxx",
        "api_secret": "xxx"
    }

    response = requests.post(url, json=data)
    token_info = response.json()
    return token_info['result']['access_token']


# 调用Assistant会话的函数（非流式）
def call_assistant_non_streaming(access_token, message):
    url = "https://chatglm.cn/chatglm/assistant-api/v1/stream_sync"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "assistant_id": "xxx",
        "prompt": message
    }
    if "conversation_id" in st.session_state:
        payload["conversation_id"] = st.session_state.conversation_id

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error calling assistant: {}".format(response.text))


def main():
    # 侧边栏模型选择
    chosen_model = st.sidebar.selectbox(
        "选择模型",
        options=list(MODEL_OPTIONS.keys()),
        index=0  # 默认选择第一个
    )

    if chosen_model == "ChatLaw-司法摘要":
        # 根据选择的模型名称获取路径
        model_path = MODEL_OPTIONS[chosen_model]
        model, tokenizer = init_model(model_path)
        if model is None or tokenizer is None:
            st.stop()

    # 初始化聊天记录
    messages = init_chat_history()

    # 显示欢迎语（仅在首次加载时）
    if not messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"您好，我是{chosen_model}，很高兴为您服务💖")

    # 显示历史消息
    for message in messages:
        avatar = "🙋‍♂️" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            # if 'rag' in message:
            #     with st.expander("🔍 查看依据的 RAG 内容（点击展开）"):
            #         for idx, rag in enumerate(message["rag"]):
            #             st.markdown(f"#### 来源 {idx + 1}: {rag['title']}")
            #             st.markdown(f"相关性评分: `{rag['score']:.4f}`")
            #             st.markdown(rag["content"])
            #             st.divider()

    # 处理用户输入
    if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
        # 添加用户消息
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        # 生成回复
        with st.chat_message("assistant", avatar="🤖"):
            try:
                if chosen_model == "ChatLaw-司法摘要":
                    # 构建模型输入
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                    # 生成文本
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                        temperature=0.95,
                        do_sample=True,
                        top_p=0.7
                    )

                    # 解码并提取新生成内容
                    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
                    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    # 流式显示
                    st.write_stream(response_generator(response))
                    messages.append({"role": "assistant", "content": response})
                else:
                    try:
                        access_token = get_access_token()
                        result = call_assistant_non_streaming(access_token, prompt)
                        # 提取最终回答
                        response = ""
                        final_output = result['result']['output']
                        if "conversation_id" not in st.session_state:
                            st.session_state.conversation_id = result['result']['conversation_id']
                        for item in final_output:
                            for content_item in item['content']:
                                if 'type' in content_item and content_item['type'] == 'text':
                                    response += content_item['text']
                        # 流式显示
                        st.write_stream(response_generator(response))
                        # rag_contents = []
                        # for item in final_output:
                        #     for content_item in item['content']:
                        #         if 'type' in content_item and content_item['type'] == 'rag_slices':
                        #             for doc in content_item['content']:
                        #                 rag_contents.append({
                        #                     "title": doc.get("document_name", "未知文件"),
                        #                     "content": doc.get("text", ""),
                        #                     "score": doc.get("score", 0)
                        #                 })

                        # # 展示 RAG 内容（可折叠）
                        # if rag_contents:
                        #     with st.expander("🔍 查看依据的 RAG 内容（点击展开）"):
                        #         for idx, rag in enumerate(rag_contents):
                        #             st.markdown(f"#### 来源 {idx + 1}: {rag['title']}")
                        #             st.markdown(f"相关性评分: `{rag['score']:.4f}`")
                        #             st.markdown(rag["content"])
                        #             st.divider()

                        messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        print(e)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    st.error("显存不足，请缩短输入或重启会话。")
                else:
                    st.error(f"生成失败: {str(e)}")

        # 清空对话按钮
        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
