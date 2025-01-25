# 必要なライブラリをインポート
import streamlit as st  # Webアプリケーションを作成するためのライブラリ
from langchain_community.chat_models import ChatOpenAI # ChatGPTのモデルを利用するためのクラス
from langchain.schema import (  # チャットの各メッセージタイプを定義するクラス
    SystemMessage,  # システムメッセージ用
    HumanMessage,   # ユーザーメッセージ用
    AIMessage       # AI(ChatGPT)からの返答用
)
from langchain.callbacks import get_openai_callback  # ChatGPT APIの使用料金を計算するためのツール


def init_page():
    """
    Streamlitのページ設定を初期化する関数
    ページのタイトル、アイコン、ヘッダーを設定
    """
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")
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
    使用するChatGPTモデルとパラメータを選択する関数
    - モデルの種類（GPT-3.5, GPT-4, GPT-4o-mini）を選択
    - AIの応答のランダム性を調整するtemperatureパラメータを設定
    """
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4", "GPT-4o-mini"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    elif model == "GPT-4":
        model_name = "gpt-4"
    elif model == "GPT-4o-mini":
        model_name = "gpt-4o-mini"
    
    # temperatureの値が高いほど、より創造的でランダムな応答になります
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)


def get_answer(llm, messages):
    """
    ChatGPTから回答を取得し、その使用コストを計算する関数
    - llm: ChatGPTモデル
    - messages: 会話履歴
    戻り値: (回答の内容, APIの使用コスト)
    """
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def main():
    """
    メインの実行関数
    - ページの初期化
    - モデルの選択
    - メッセージの初期化
    - チャットインターフェースの表示と対話の処理
    - コストの表示
    """
    init_page()

    llm = select_model()
    init_messages()

    # ユーザーからの入力を受け付け、ChatGPTからの回答を取得
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # チャット履歴の表示
    # SystemMessage, HumanMessage, AIMessageの種類に応じて適切な形式で表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    # サイドバーにAPIの使用コストを表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

# プログラムのエントリーポイント
if __name__ == '__main__':
    main()