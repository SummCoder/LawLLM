import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from PyPDF2 import PdfReader
from docx import Document
import os
import random

st.set_page_config(page_title="LawLLM-司法摘要", layout="wide")
st.title("📑 法律文书智能摘要系统")

# 预定义配置
MODEL_PATH = "D:\\models\\Qwen2.5-0.5B-sfzy"
SUPPORTED_FORMATS = ["pdf", "docx", "txt"]
MAX_FILE_SIZE = 10  # MB


@st.cache_resource
def init_model():
    """初始化模型"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None


def parse_uploaded_file(uploaded_file):
    """解析上传文件"""
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == "pdf":
        return parse_pdf(uploaded_file)
    elif file_type == "docx":
        return parse_docx(uploaded_file)
    elif file_type == "txt":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        raise ValueError("不支持的文档格式")


def parse_pdf(uploaded_file):
    """解析PDF文件"""
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF解析失败: {str(e)}")
        return ""


def parse_docx(uploaded_file):
    """解析Word文档"""
    try:
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"DOCX解析失败: {str(e)}")
        return ""


def generate_summary(model, tokenizer, text):
    """生成文书摘要"""
    try:
        # 构建提示词，随机选择一个提示词进行拼接
        prompt = random.choice(
            [
                f"""请归纳这篇文书的大致要点。\n{text}""",
                f"""以下是一篇法律文书：\n{text}\n请大致描述这篇文书的内容。""",
                f"""请对这篇法律文书进行摘要\n\n{text}""",
                f"""{text}\n以上是一篇法律文书，请归纳这篇文书的大致要点。"""
            ]
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=4096,
            truncation=True
        ).to(model.device)

        # 生成参数
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.95,
            do_sample=True,
            top_p=0.7,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        # 提取生成内容
        summary = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        return summary
    except Exception as e:
        st.error(f"摘要生成失败: {str(e)}")
        return ""


def main():
    # 初始化模型
    model, tokenizer = init_model()
    if model is None or tokenizer is None:
        st.stop()

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统设置")
        uploaded_file = st.file_uploader(
            "上传法律文书",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=False,
            help=f"支持格式：{', '.join(SUPPORTED_FORMATS)} | 最大文件：{MAX_FILE_SIZE}MB",
        )

        # 文件大小校验
        if uploaded_file and (uploaded_file.size > MAX_FILE_SIZE * 1024 * 1024):
            st.error(f"文件大小超过限制（最大 {MAX_FILE_SIZE}MB）")
            st.stop()

    # 主界面
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📥 原文内容")
        if uploaded_file:
            with st.spinner("正在解析文档..."):
                raw_text = parse_uploaded_file(uploaded_file)
                if raw_text:
                    st.info(f"文档解析成功（长度：{len(raw_text)}字符）")
                    with st.expander("查看原文内容", expanded=False):
                        st.text(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))
                else:
                    st.warning("文档内容为空")

    with col2:
        st.subheader("📝 智能摘要")
        if uploaded_file and raw_text:
            start_analysis = st.button("开始生成摘要", type="primary")

            if start_analysis:
                with st.spinner("正在生成专业摘要..."):
                    start_time = time.time()
                    summary = generate_summary(model, tokenizer, raw_text)
                    elapsed = time.time() - start_time

                    st.success(f"摘要生成完成（耗时：{elapsed:.1f}s）")
                    st.markdown("### 摘要结果")
                    st.write_stream(summary_generator(summary))

                    # 添加下载按钮
                    st.download_button(
                        label="下载摘要",
                        data=summary,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_摘要.txt",
                        mime="text/plain"
                    )


def summary_generator(text):
    """摘要流式生成效果"""
    for word in text:
        yield word
        time.sleep(0.02)


if __name__ == "__main__":
    main()