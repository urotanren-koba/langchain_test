# 必要なライブラリのインポート
import streamlit as st  # Webアプリケーション作成用のライブラリ
from langchain.chat_models import ChatOpenAI  # ChatGPTを利用するためのクラス
from langchain.callbacks import get_openai_callback  # OpenAI APIの使用料金計算用

from langchain.prompts import PromptTemplate  # プロンプトテンプレート作成用
from langchain.chains.summarize import load_summarize_chain  # 要約チェーン作成用
from langchain.document_loaders import YoutubeLoader  # Youtube動画の字幕取得用
import yt_dlp  # Youtube動画のメタデータ取得用


def init_page():
    """
    Streamlitのページ設定を初期化する関数
    """
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")
    st.session_state.costs = []  # APIコスト記録用の配列を初期化


def select_model():
    """
    使用するGPTモデルを選択する関数
    サイドバーでGPT-3.5、GPT-4、GPT-4o-miniから選択可能
    """
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4","GPT-4o-mini"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    elif model == "GPT-4":
        model_name = "gpt-4"
    elif model == "GPT-4o-mini":
        model_name = "gpt-4o-mini"

    return ChatOpenAI(temperature=0, model_name=model_name)


def get_url_input():
    """
    YoutubeのURLを入力するテキストボックスを表示する関数
    """
    url = st.text_input("Youtube URL: ", key="input")
    return url


def get_document(url):
    """
    指定されたYoutube URLから字幕を取得する関数
    日本語→英語の順で字幕を探し、見つかった方を返す
    """
    with st.spinner("Fetching Content ..."):
        try:
            # yt-dlpの設定
            ydl_opts = {
                'writesubtitles': True,  # 字幕を取得
                'writeautomaticsub': True,  # 自動生成字幕も取得
                'subtitleslangs': ['ja', 'en'],  # 日本語と英語の字幕を取得
                'skip_download': True,  # 動画本体はダウンロードしない
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                # まず手動で追加された字幕を探す
                if 'subtitles' in info and info['subtitles']:
                    for lang in ['ja', 'en']:
                        if lang in info['subtitles']:
                            caption_url = info['subtitles'][lang][0]['url']
                            import requests
                            response = requests.get(caption_url)
                            subtitle_content = response.text
                            from langchain.schema import Document
                            return [Document(
                                page_content=subtitle_content,
                                metadata={"source": url, "title": info.get('title', 'Unknown')}
                            )]

                # 手動字幕がない場合は自動字幕を探す
                if 'automatic_captions' in info and info['automatic_captions']:
                    for lang in ['ja', 'en']:
                        if lang in info['automatic_captions']:
                            caption_url = info['automatic_captions'][lang][0]['url']
                            import requests
                            response = requests.get(caption_url)
                            subtitle_content = response.text
                            from langchain.schema import Document
                            return [Document(
                                page_content=subtitle_content,
                                metadata={"source": url, "title": info.get('title', 'Unknown')}
                            )]

                raise Exception('No captions or automatic subtitles available')

        except Exception as e:
            st.error(f"Failed to fetch video content: {str(e)}")
            return None


def summarize(llm, docs):
    """
    取得した字幕を要約する関数
    LangChainの要約チェーンを使用して字幕を要約する
    """
    # 要約用のプロンプトテンプレート
    prompt_template = """以下はYoutube動画の字幕です。この内容を要約してください。

要約のガイドライン:
1. 動画の主なトピックやテーマを明確に説明する
2. 重要なポイントやハイライトを含める
3. 具体的な例やエピソードがあればそれも含める
4. 簡潔でわかりやすい日本語で書く

字幕内容:
============

{text}

============

上記のガイドラインに従って、300文字以内で要約してください:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # OpenAI APIの使用料金を計算しながら要約を実行
    with get_openai_callback() as cb:
        chain = load_summarize_chain( 
            llm,
            chain_type="stuff",  # 全テキストを一度に要約
            verbose=True,
            prompt=PROMPT
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
        
    return response['output_text'], cb.total_cost


def main():
    """
    メインの実行関数
    アプリケーションの全体の流れを制御する
    """
    init_page()
    llm = select_model()

    # 入力用と出力用のコンテナを作成
    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        if url:
            document = get_document(url)
            if document:  # documentがNoneでない場合のみ処理を続ける
                with st.spinner("ChatGPT is typing ..."):
                    output_text, cost = summarize(llm, document)
                st.session_state.costs.append(cost)
            else:
                output_text = None
                st.error("Failed to process the video. Please check if the video has captions available.")
        else:
            output_text = None

    # 要約結果と元のテキストを表示
    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

    # サイドバーにAPIの使用コストを表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()