import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from io import BytesIO
import os, sys, json


@st.cache_resource
def get_openai_client(url, api_key):
    client = OpenAI(base_url=url, api_key=api_key)
    return client


def stt_page():
    st.title("Speech to Text(语音转文字)")
    st.caption("use Whisper to convert speech to text")

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
    if 'button_active' not in st.session_state:
        st.session_state.button_active = False
    else:
        button_active = st.session_state.button_active

    # 获取当前脚本文件的绝对路径吗，并提取目录路径
    src_path = os.path.dirname(os.path.relpath(sys.argv[0]))
    with open(os.path.join(src_path, 'config/default.json'), 'r', encoding='utf-8') as f:
        config_default = json.load(f)

    client = get_openai_client(base_url, api_key)
    audio_file = None

    # 选择输入方式，录音或者上传
    option = st.radio("请选择输入方式:", ("Recording", "Uploading"), horizontal=True, index=0)
    if option == "Recording":
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.text("录音音频")
            st.audio(audio_bytes, format="audio/wav")
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            st.session_state.button_active = True
    else:
        st.warning("注意，当前OpenAI接口最大仅支持25MB！")
        audio_file = st.file_uploader("上传音频文件", type=["wav", "mp3", "m4a"])
        if audio_file:
            if audio_file.size > MAX_FILE_SIZE:
                st.error("文件大小超过最大限制--25MB，请重新上传！")
                st.stop()
            else:
                st.text("上传的音频文件")
                st.audio(audio_file, format="audio/wav")
                st.session_state.button_active = True

    # 定义一个函数 whisper_online,用于在线调用Whisper模型进行语音转文字
    def whisper_online(audio_file):
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="中文"
        )
        return transcript

    button_active = not st.session_state.button_active
    if st.button("开始转换", disabled=button_active):
        if audio_file:
            with  st.spinner("请等待···"):
                try:
                    transcript = whisper_online(audio_file)
                    st.success("转换完成")
                    st.write(transcript)
                except Exception as e:
                    st.error(e)
                    st.stop()
        else:
            st.warning("Please upload the audio file first.")


MAX_FILE_SIZE = 25 * 1024 * 1024

if __name__ == '__main__':
    stt_page()
