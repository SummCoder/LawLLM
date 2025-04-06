import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

st.set_page_config(page_title="ChatLaw")
st.title("ChatLaw")

# 预定义的模型列表（名称: 路径）
MODEL_OPTIONS = {
    "ChatLaw-法律咨询": "D:\\models\\ChatLaw-13B",
    "LaWGPT-通用法律模型": "D:\\models\\LaWGPT-7B",
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


def main():
    # 侧边栏模型选择
    chosen_model = st.sidebar.selectbox(
        "选择模型",
        options=list(MODEL_OPTIONS.keys()),
        index=0  # 默认选择第一个
    )

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

    # 处理用户输入
    if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
        # 添加用户消息
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        # 生成回复
        with st.chat_message("assistant", avatar="🤖"):
            try:
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

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    st.error("显存不足，请缩短输入或重启会话。")
                else:
                    st.error(f"生成失败: {str(e)}")

        # 清空对话按钮
        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()