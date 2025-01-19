# 必要なライブラリのインポート
# pycryptodomeはセキュリティ関連のライブラリ
from glob import glob  # ファイル検索用
import streamlit as st  # WebアプリケーションのUI作成用
from langchain.chat_models import ChatOpenAI  # ChatGPTモデルを使用するためのクラス
from langchain.llms import OpenAI  # OpenAI APIを使用するためのクラス
from langchain.callbacks import get_openai_callback  # OpenAI APIの使用料金を計算するためのツール

# PDF関連のライブラリ
from PyPDF2 import PdfReader  # PDFファイルを読み込むためのライブラリ
from langchain.embeddings.openai import OpenAIEmbeddings  # テキストをベクトル化するためのクラス
from langchain.text_splitter import RecursiveCharacterTextSplitter  # テキストを適切な長さに分割するためのクラス
from langchain.vectorstores import Qdrant  # ベクトルデータベースを扱うためのクラス
from langchain.chains import RetrievalQA  # 質問応答システムを構築するためのクラス

# Qdrant（ベクトルデータベース）関連のライブラリ
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ベクトルデータベースの保存場所とコレクション名の設定
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"


def init_page():
    """
    Streamlitのページ設定を初期化する関数
    ページのタイトル、アイコン、サイドバーを設定し、コスト計算用の配列を初期化
    """
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🤗"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []


def select_model():
    """
    使用するChatGPTモデルを選択する関数
    各モデルの最大トークン数を設定し、適切なモデルを返す
    """
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4", "GPT-4o-mini"))
    
    # モデル名とトークン数の設定
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
        st.session_state.max_token = 4096 - 300  # 余裕を持たせるため300トークン減らす
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
        st.session_state.max_token = 16384 - 300
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4"
        st.session_state.max_token = 8192 - 300
    elif model == "GPT-4o-mini":
        st.session_state.model_name = "gpt-4o-mini"
        st.session_state.max_token = 8192 - 300
    
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_pdf_text():
    """
    PDFファイルをアップロードし、テキストを抽出する関数
    抽出したテキストを適切な長さに分割して返す
    """
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',
        type='pdf'
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            # テキストを500文字ごとに分割（文脈を保持しつつ、適切な長さに分割）
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():
    """
    Qdrantベクトルデータベースを初期化・読み込む関数
    コレクションが存在しない場合は新規作成する
    """
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(pdf_text):
    """
    PDFから抽出したテキストをベクトルデータベースに保存する関数
    """
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)


def build_qa_model(llm):
    """
    質問応答モデルを構築する関数
    ベクトルデータベースから関連する情報を検索し、回答を生成する
    """
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        search_type="similarity",  # 類似度検索を使用
        search_kwargs={"k":10}  # 上位10件の関連文書を取得
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )


def page_pdf_upload_and_build_vector_db():
    """
    PDFアップロード用のページを表示する関数
    アップロードされたPDFをベクトルデータベースに保存する
    """
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


def ask(qa, query):
    """
    ユーザーの質問に対して回答を生成する関数
    APIの使用コストも計算する
    """
    with get_openai_callback() as cb:
        answer = qa(query)
    return answer, cb.total_cost


def page_ask_my_pdf():
    """
    質問応答ページを表示する関数
    ユーザーの入力を受け付け、回答を表示する
    """
    st.title("Ask My PDF(s)")

    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


def main():
    """
    メインの実行関数
    ページの初期化、ナビゲーション、コスト表示を行う
    """
    init_page()

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()

    # APIの使用コストを表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()