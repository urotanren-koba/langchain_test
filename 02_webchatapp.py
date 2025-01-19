# 必要なライブラリのインポート
import streamlit as st  # Webアプリケーションを作成するためのライブラリ
from streamlit_chat import message  # チャットUIを実装するためのコンポーネント
from langchain.chat_models import ChatOpenAI  # ChatGPTを利用するためのクラス
from langchain.schema import (  # チャットの各メッセージタイプを定義するクラス
    SystemMessage,  # システムメッセージ用
    HumanMessage,   # ユーザーメッセージ用
    AIMessage       # AI(ChatGPT)からの返答用
)
from langchain.callbacks import get_openai_callback  # ChatGPT APIの使用料金を計算するためのツール

# Webスクレイピングに必要なライブラリ
import requests  # HTTPリクエストを送信するためのライブラリ
from bs4 import BeautifulSoup  # HTMLを解析するためのライブラリ
from urllib.parse import urlparse  # URLを解析するためのライブラリ


def init_page():
    """
    Streamlitのページ設定を初期化する関数
    ページのタイトル、アイコン、ヘッダーを設定
    """
    st.set_page_config(
        page_title="Webサイト要約",
        page_icon="🤗"
    )
    st.header("Webサイト要約 🤗")
    st.sidebar.title("Options")


def init_messages():
    """
    チャットの会話履歴を初期化する関数
    - サイドバーに会話をクリアするボタンを設置
    - 初回アクセス時または会話クリア時にシステムメッセージを設定
    """
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model():
    """
    使用するChatGPTモデルを選択する関数
    GPT-3.5, GPT-4, GPT-4o-miniから選択可能
    """
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4", "GPT-4o-mini"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    elif model == "GPT-4":
        model_name = "gpt-4"
    elif model == "GPT-4o-mini":
        model_name = "gpt-4o-mini"

    return ChatOpenAI(temperature=0, model_name=model_name)


def get_url_input():
    """
    ユーザーからURLを入力として受け取る関数
    """
    url = st.text_input("URL: ", key="input")
    return url


def validate_url(url):
    """
    入力されたURLが有効なものかチェックする関数
    スキーム(http/https)とドメインが存在するかを確認
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    """
    指定されたURLからWebページの内容を取得する関数
    main要素、article要素、またはbody要素からテキストを抽出
    """
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # 優先順位: main > article > body の順でテキストを取得
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write('something wrong')
        return None


def build_prompt(content, n_chars=300):
    """
    ChatGPTに送信するプロンプトを生成する関数
    Webページの内容を指定された文字数で要約するように指示
    """
    return f"""以下はとある。Webページのコンテンツである。内容を{n_chars}程度でわかりやすく要約してください。

========

{content[:1000]}

========

日本語で書いてね！
"""


def get_answer(llm, messages):
    """
    ChatGPTから回答を取得し、その使用コストを計算する関数
    """
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def main():
    """
    メインの実行関数
    - ページの初期化
    - モデルの選択
    - URLの入力受付
    - コンテンツの取得と要約
    - 結果の表示
    - コストの表示
    """
    init_page()

    llm = select_model()
    init_messages()

    # 入力用と出力用のコンテナを作成
    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_prompt(content)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None

    # 要約結果と元のテキストを表示
    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)

    # サイドバーにAPIの使用コストを表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()