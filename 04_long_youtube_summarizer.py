import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yt_dlp
from langchain.schema import Document
import requests

def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")
    
    # 初期化
    if "costs" not in st.session_state:
        st.session_state.costs = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    if "max_token" not in st.session_state:
        st.session_state.max_token = 16384 - 300  # gpt-3.5-turboのデフォルト値


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    
    # モデル名の設定
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
        st.session_state.max_token = 16384 - 300
    else:
        st.session_state.model_name = "gpt-4"
        st.session_state.max_token = 8192 - 300
    
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url


def get_document(url):
    if not url:
        return None
        
    with st.spinner("Fetching Content ..."):
        try:
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['ja', 'en'],
                'skip_download': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # まず手動で追加された字幕を探す
                if 'subtitles' in info and info['subtitles']:
                    for lang in ['ja', 'en']:
                        if lang in info['subtitles']:
                            caption_url = info['subtitles'][lang][0]['url']
                            response = requests.get(caption_url)
                            subtitle_content = response.text
                            
                            # テキスト分割
                            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                                chunk_size=st.session_state.max_token,
                                chunk_overlap=0,
                            )
                            texts = text_splitter.split_text(subtitle_content)
                            
                            return [Document(
                                page_content=text,
                                metadata={"source": url, "title": info.get('title', 'Unknown')}
                            ) for text in texts]

                # 手動字幕がない場合は自動字幕を探す
                if 'automatic_captions' in info and info['automatic_captions']:
                    for lang in ['ja', 'en']:
                        if lang in info['automatic_captions']:
                            caption_url = info['automatic_captions'][lang][0]['url']
                            response = requests.get(caption_url)
                            subtitle_content = response.text
                            
                            # テキスト分割
                            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                                chunk_size=st.session_state.max_token,
                                chunk_overlap=0,
                            )
                            texts = text_splitter.split_text(subtitle_content)
                            
                            return [Document(
                                page_content=text,
                                metadata={"source": url, "title": info.get('title', 'Unknown')}
                            ) for text in texts]

                raise Exception('No captions or automatic subtitles available')

        except Exception as e:
            st.error(f"Failed to fetch video content: {str(e)}")
            return None


def summarize(llm, docs):
    prompt_template = """以下はYouTube動画の字幕から抽出されたテキストの一部です。
このテキストの内容を要約してください。

テキスト:
{text}

日本語で要約してください。
以下のポイントを含めてください：
1. 主なトピックや話題
2. 重要なポイントや発言
3. 具体的な例示やエピソード（あれば）
"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            verbose=True,
            map_prompt=PROMPT,
            combine_prompt=PROMPT
        )
        response = chain(
            {
                "input_documents": docs,
                "token_max": st.session_state.max_token
            },
            return_only_outputs=True
        )
        
    return response['output_text'], cb.total_cost


def main():
    init_page()
    llm = select_model()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        if url:
            document = get_document(url)
            if document:
                with st.spinner("ChatGPT is typing ..."):
                    output_text, cost = summarize(llm, document)
                st.session_state.costs.append(cost)
            else:
                output_text = None
        else:
            output_text = None

    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()