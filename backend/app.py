import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


st.set_page_config(page_title="LawLLM")
st.title("LawLLM")


@st.cache_resource
def init_model():
    model_path = "D:\\models\\Qwen2.5-0.5B-sfzy"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto"
    )
    if torch.cuda.is_available():
        model.to('cuda')  # 显式地将模型放到GPU上

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("您好，我是司法摘要大模型，很高兴为您服务💖")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🙋‍♂️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()
    messages = init_chat_history()
    if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            placeholder.markdown(response)
        messages.append({"role": "assistant", "content": response})
        # print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()