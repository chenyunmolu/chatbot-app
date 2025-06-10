import base64
import io
import json
import os
import sys
import time

import tiktoken
from openai import OpenAI
import requests
import streamlit as st
from PIL import Image


# 聊天界面
def chat_page():
    st.title("Chat（文本对话聊天）")
    # 初始化参数
    if 'base_url' not in st.session_state:
        st.session_state.base_url = os.getenv("OPENAI_BASE_URL")
        base_url = os.getenv("OPENAI_BASE_URL")
    else:
        base_url = st.session_state.base_url

    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv("OPENAI_API_KEY")
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = st.session_state.api_key

    if (api_key is None) or (base_url == ''):
        st.warning("请先设置API_KEY")
        st.stop()

    src_path = os.path.dirname(os.path.relpath(sys.argv[0]))
    with  open(os.path.join(src_path, 'config/default.json'), 'r') as f:
        config_default = json.load(f)

    st.session_state['model_list'] = config_default['completions']['models']
    model_name = st.selectbox("Models", st.session_state.model_list, key="chat_model_name")

    option = st.radio("system_prompt", ("Manual input", "prompts"), horizontal=True, index=0)
    if option == "Manual input":
        system_prompt = st.text_input('System Prompt (Please click the button "clear history" after modification.)',
                                      config_default["completions"]["system_prompt"])
    else:
        with  open(os.path.join(src_path, 'config/prompt.json'), 'r', encoding='utf-8') as f:
            masks = json.load(f)
        masks_zh = [item['name'] for item in masks['zh']]
        masks_zh_name = st.selectbox("prompts", masks_zh)
        for item in masks['zh']:
            if item['name'] == masks_zh_name:
                system_prompt = item['context']
                break

    # 是否使用默认参数
    if not st.checkbox("Use default parameters", value=True):
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=200000,
                                     value=config_default['completions']['max_tokens'], key="max_tokens")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0,
                                value=config_default['completions']['temperature'], key="temperature")
        top_p = st.slider("top_p", min_value=0.0, max_value=1.0,
                          value=config_default['completions']['top_p'], key="top_p")
        stream = st.checkbox("stream", value=config_default['completions']['stream'], key="stream")
    else:
        max_tokens = config_default['completions']['max_tokens']
        temperature = config_default['completions']['temperature']
        top_p = config_default['completions']['top_p']
        stream = config_default['completions']['stream']

    # 初始化聊天记录
    if 'chat_messages' not in st.session_state:
        st.session_state['chat_messages'] = [
            {"role": "system", "content": system_prompt}
        ]
    # 清除历史记录
    if st.button("clear history"):
        st.session_state['chat_messages'] = [
            {"role": "system", "content": system_prompt}
        ]
        st.info("History cleared.")
    # 显示聊天记录
    for msg in st.session_state['chat_messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理用户输入
    if prompt := st.chat_input():
        try:
            client = get_openai_client(base_url, api_key)
        except  Exception as e:
            st.error(e)
            st.stop()
        # 显示用户的输入内容
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                try:
                    temp_chat_messages = st.session_state['chat_messages']
                    temp_chat_messages.append({"role": "user", "content": prompt})
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=temp_chat_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=stream
                    )
                except Exception as e:
                    st.error(e)
                    st.stop()

                if response:
                    if stream:
                        placeholder = st.empty()
                        streaming_text = ''
                        stop_button = st.button("⏹️ 停止流式输出")  # 添加停止按钮

                        for chunk in response:
                            if stop_button or chunk.choices[0].finish_reason == "stop":
                                break
                            try:
                                chunk_text = chunk.choices[0].delta.content or ""
                            except AttributeError:
                                chunk_text = ""

                            streaming_text += chunk_text
                            placeholder.markdown(
                                f"<div style='font-family: sans-serif;'>{streaming_text}</div>",
                                unsafe_allow_html=True
                            )
                        model_msg = streaming_text
                    else:
                        model_msg = response.choices[0].message.content
                        st.markdown(model_msg)
                    end_time = time.time()
                    temp_chat_messages.append({"role": "assistant", "content": model_msg})
                    st.session_state['chat_messages'] = temp_chat_messages

                    # 计算当前对话的消耗的token数
                    if config_default["completions"]["num_tokens"]:
                        try:
                            # 调用函数计算 token 数量
                            num_tokens = num_tokens_from_messages(st.session_state.chat_messages, model=model_name)
                            # 显示 token 数量信息
                            info_num_tokens = f"use tokens: {num_tokens}"
                            st.info(info_num_tokens)
                        except Exception as e:
                            print(e)
                    # 生成当前对话耗时信息
                    if config_default["completions"]["use_time"]:
                        st.info(f"Use time: {round(end_time - start_time, 2)}s")


def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
    }:
        # 每个消息有一个基本的令牌数 tokens_per_message，默认3个token，每个 name 属性预设的固定令牌数 tokens_per_name，假设其值为 1。
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    # 函数通过迭代消息列表，并根据消息的角色 (如 user、assistant、tool、system) 计算令牌数量。
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # 每个回复都以 <|start|>assistant<|message|> 开头
    # 例如：<|start|>assistant<|message|>今天天气很好，适合出门！ <|end|>
    num_tokens += 3
    return num_tokens


@st.cache_resource
def get_openai_client(base_url, api_key):
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client


if __name__ == '__main__':
    chat_page()
