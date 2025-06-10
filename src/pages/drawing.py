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


@st.cache_resource
def get_openai_client(url, api_key):
    '''
    使用了缓存，当参数不变时，不会重复创建client
    '''
    client = OpenAI(base_url=url, api_key=api_key)
    return client


def drawing_page():
    st.title("Drawing(文生图)")
    st.caption("use DALLE3 to draw images")
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
    image_size = st.selectbox("选择图片大小", ["1024x1024", "1024x1792", "1792x1024"], key="image_size")
    quality = st.selectbox("选择图片质量", ["standard", "hd"], key="quality")
    num_images = st.selectbox('图片数量 （dall-e-3 only n=1）', [1], key='num_images')

    if prompt := st.chat_input("请输入你的描述"):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = get_openai_client(base_url, api_key)
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size=image_size,
                        quality=quality,
                        n=num_images
                    )

                    for image in response.data:
                        image_url = image.url
                        revised_url = image.revised_prompt
                        st.image(image_url, caption=revised_url, width=300)
                        # 添加下载链接
                        download_link = f'<a href="{image_url}" download>Download Image</a>'
                        st.markdown(download_link, unsafe_allow_html=True)
                        st.write(f"Revised Prompt: {revised_url}")
                except Exception as e:
                    st.error(e)
                    st.stop()


if __name__ == '__main__':
    drawing_page()
