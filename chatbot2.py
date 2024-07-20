from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV 파일 불러오기
file_path = 'crawling_divorce_cleaned_v3.csv'
data = pd.read_csv(file_path)

# 결측값 제거
data_cleaned = data.dropna()

# 중복 데이터 제거
data_cleaned = data_cleaned.drop_duplicates()

# 필요 없는 열 제거 (판례일련번호는 필요 없을 수 있음)
data_cleaned = data_cleaned[['사건명', '사건번호', '법원명', '판결요지', '판례내용']]

# 데이터 확인
print(data_cleaned.head())

# HuggingFaceEmbeddings 모델 로드
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB 클라이언트 생성
persist_directory = "./chroma"

# 디렉토리 생성
os.makedirs(persist_directory, exist_ok=True)

try:
    client = PersistentClient(path=persist_directory)
    print(f"ChromaDB PersistentClient created with persist directory: {persist_directory}")
except Exception as e:
    print(f"Failed to create ChromaDB PersistentClient: {e}")

# 컬렉션 생성 또는 가져오기
def get_or_create_collection(client, collection_name):
    try:
        collection = client.get_or_create_collection(name=collection_name)
        print(f"Collection '{collection_name}' retrieved or created.")
        return collection
    except Exception as e:
        print(f"Failed to retrieve or create collection: {e}")
        return None

collection = get_or_create_collection(client, "legal_cases")

if collection:
    # 문서를 청크로 나누는 함수
    def split_documents(data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = []
        for _, row in data.iterrows():
            doc = Document(page_content=row['판례내용'], metadata={
                "사건명": row['사건명'],
                "사건번호": row['사건번호'],
                "법원명": row['법원명'],
                "판결요지": row['판결요지']
            })
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        return chunks

    chunks = split_documents(data_cleaned)
    print(f"Total chunks created: {len(chunks)}")

    # 임베딩된 문서들을 ChromaDB에 저장합니다.
    def add_chunks_to_collection(collection, chunks, batch_size=10):
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            try:
                batch_embeddings = embeddings.embed_documents([chunk.page_content for chunk in batch_chunks])
                for idx, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    collection.add(
                        ids=[str(i + idx)],
                        documents=[chunk.page_content],
                        embeddings=[embedding],
                        metadatas=[chunk.metadata]
                    )
                    logger.info(f"Chunk {i + idx} added to collection.")
            except Exception as e:
                logger.error(f"Failed to add chunks {i} to {i+batch_size}: {e}")
                time.sleep(5)  # 재시도 전 잠시 대기

    add_chunks_to_collection(collection, chunks)
    logger.info("Data has been embedded and added to the ChromaDB collection.")

    # 저장된 데이터 확인
    def verify_stored_data(collection, embeddings, sample_text):
        query_embedding = embeddings.embed_query(sample_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        print("Stored Documents:")
        for doc in results["documents"]:
            print(doc)
        print("Metadata for stored documents:")
        for meta in results["metadatas"]:
            print(meta)

    verify_stored_data(collection, embeddings, chunks[0].page_content)

    # chroma 디렉토리 내 파일을 출력합니다.
    if os.path.exists(persist_directory):
        print("ChromaDB directory contents:")
        for root, dirs, files in os.walk(persist_directory):
            for file in files:
                print(os.path.join(root, file))
    else:
        print("ChromaDB directory does not exist.")
else:
    print("Collection could not be created or retrieved.")

# ChromaDB에서 데이터를 검색할 수 있도록 Retriever를 설정합니다.
def chroma_retriever(query, collection, embeddings, n_results=3):
    query_embedding = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    # 검색된 문서 출력
    print("Retrieved Documents:")
    for idx, doc in enumerate(results["documents"]):
        print(f"Document {idx+1}: {doc}")
    print("\nMetadata for retrieved documents:")
    for idx, meta in enumerate(results["metadatas"]):
        print(f"Metadata {idx+1}: {meta}")
    
    return results["documents"]

# OpenAI API 키 설정
openai_api_key = ""

# LangChain 설정
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

def get_answer(question, context_messages, n_results=3):
    # 질문을 임베딩하여 ChromaDB에서 유사도가 높은 문서를 검색합니다.
    docs = chroma_retriever(question, collection, embeddings, n_results)
    context = "\n\n".join([f"Document {idx+1}: {doc}" for idx, doc in enumerate(docs)])
    
    # OpenAI 모델의 최대 컨텍스트 길이를 초과하지 않도록 자릅니다.
    max_context_length = 4097 - 256  # 최대 토큰 길이에서 여유를 둡니다.
    if len(context) > max_context_length:
        context = context[:max_context_length]
    
    messages = context_messages + [
        {"role": "user", "content": f"Question: {question}\n\nContext: {context}\n\nAnswer:"}
    ]
    
    response = llm.invoke(messages)
    return response.content, messages

# 멀티턴 대화
context_messages = [
    {"role": "system", "content": "You are an AI assistant trained to provide legal information based on the given context. Use the following documents to answer the question as accurately as possible."}
]

while True:
    question = input("질문을 입력하세요 (끝내려면 '종료' 입력): ")
    if question.lower() == "종료":
        break
    answer, context_messages = get_answer(question, context_messages, n_results=5)
    print(answer)
