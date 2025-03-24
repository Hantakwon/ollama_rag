import os
from glob import glob
import re
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# PDF 파일 로드
pdf_files = glob(os.path.join('pdf_data', '스플릿 연습.pdf'))
loader = PyPDFLoader(pdf_files[0])
data = loader.load()

# 사례별 텍스트 패턴 정의
split_pattern = r'사례\s\d+.*?사건번호\s[\w\d]+\s?\|\s?결정일자\s\d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.'

# 사건 메타데이터 추출을 위한 Pydantic 모델 정의
class Case(BaseModel):
    case_number: str = Field(description="The number of the case")
    case_date: str = Field(description="The date when the case occurred (YYYY.MM.DD)")

# Ollama 기반 LLM 설정
llm = ChatOllama(model="mistral", format="json", temperature=0)

# 사건 정보 추출을 위한 프롬프트 설정
prompt = PromptTemplate.from_template("""
Extract relevant information from the following text:

TEXT: {text} \n
AI: """)

# LLM 기반 Extraction Chain 생성
runnable = prompt | llm.with_structured_output(schema=Case)

# PDF 문서에서 사례별 텍스트 추출 및 메타데이터 저장
pdf_docs = []
case_metadata_list = []

for doc in data:
    split_texts = re.findall(split_pattern, "".join(doc.page_content))
    if not split_texts:
        continue  # 패턴이 없는 경우 스킵
    
    for split_text in split_texts:
        case_metadata = {}
        
        # 사건 ID 추출
        case_id_match = re.search(r'사례\s?(\d+)\D*사건번호', split_text)
        if case_id_match:
            case_metadata['case_id'] = case_id_match.group(1)
        
        # 사건 제목과 본문 분리
        parts = re.split(split_pattern, "".join(doc.page_content))
        if len(parts) > 1:
            search_result = re.search(r'주문', parts[1])
            if search_result:
                case_metadata['title'] = parts[1][:search_result.span()[0]].strip()
                doc.page_content = parts[1][search_result.span()[0]:].strip()
            else:
                case_metadata['title'] = ''
        
        # LLM을 사용하여 사건 정보 추출
        for _ in range(10):
            try:
                response = runnable.invoke({"text": split_text})
                for k, v in dict(response).items():
                    case_metadata[k] = v.replace("\n", "").replace(" ", "")
                break
            except:
                continue
        
        doc.metadata.update(case_metadata)
        case_metadata_list.append(case_metadata)
    
    pdf_docs.append(doc)

# HuggingFace Embeddings 모델 설정
embeddings_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# Tokenizer 로드 및 문서 분할
tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1024,
    chunk_overlap=20,
)
split_docs = text_splitter.split_documents(pdf_docs)

# 분할된 문서에 제목 포함
final_docs = []
for doc in split_docs:
    title = doc.metadata.get('title', '제목 없음')
    doc.page_content = f"### 이 사건은 '{title}'에 대한 사례입니다.\n\n" + re.sub(r'(?<!\.)\n', ' ', doc.page_content)
    final_docs.append(doc)

# 벡터 데이터베이스 생성 및 저장
vectorstore = Chroma.from_documents(
    documents=final_docs,
    embedding=embeddings_model,
    collection_name="consumer_case_qa",
    persist_directory="./chroma_db"
)

# RAG 기반 질의응답 프롬프트 설정
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt2 = ChatPromptTemplate.from_template(template)

# RAG Chain 생성
llm2 = ChatOllama(model="qwen2", temperature=0)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 검색을 위한 질의
query = "배송 중 물품파손 손해배상은 어떻게 이루어지나요?"

retriever = vectorstore.as_retriever(
    search_kwargs={
        'k': 5  # 상위 5개 문서를 검색
    }
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt2
    | llm2
    | StrOutputParser()
)

# 최종 응답 실행
response = rag_chain.invoke(query)
print("응답 :", response)
