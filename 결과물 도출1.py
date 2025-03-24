import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from pprint import pprint
import re
from pydantic import BaseModel, Field  # ✅ Pydantic v2 사용
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama  # ✅ Ollama 최신 모듈 사용
# HugoingFace Embeddings를 다운로드
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# pdf 파일 경로
pdf_files = glob(os.path.join('pdf_data', '스플릿 연습.pdf'))
# print(pdf_files)

# 가져온 pdf 읽어오기
loader = PyPDFLoader(pdf_files[0])
data = loader.load()

# 2번째 페이지 내용 가져오기
# pprint(data[1].page_content)

split_pattern = r'사례\s\d+.*?사건번호\s[\w\d]+\s?\|\s?결정일자\s\d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.'
split_text = re.findall(split_pattern, "".join(data[1].page_content))
# print(split_text)

# split_text가 존재하면 split_pattern으로 분리
if split_text:
    parts = re.split(split_pattern, "".join(data[1].page_content))

# 분리된 텍스트 출력
# print(parts[1])

# 특정 문자열의 위치 찾기
search_result = re.search(r'주문', parts[1])
# 제목 찾기
title = parts[1][:search_result.span()[0]].strip()
# 내용 찾기
content = parts[1][search_result.span()[0]:]

# 구분 문자열에서 사례 번호 추출
print(split_text[0])
print(re.findall(r'사례\s?(\d+)\D*사건번호', split_text[0])[0])

# ✅ 사건에 대한 메타데이터 추출 (Pydantic v2 적용)
class Case(BaseModel):
    case_number: str = Field(description="The number of the case")
    case_date: str = Field(description="The date when the case occurred (YYYY.MM.DD)")

# ✅ Extraction chain 구성
prompt = PromptTemplate.from_template(
    """ Extract relevant information from the following text:
    
TEXT: {text} \n

AI: """
)

text = split_text[0]

# ✅ 최신 Ollama 모듈 사용
llm = ChatOllama(model="mistral", format="json", temperature=0)

# ✅ Pydantic v2 대응 (`schema()` 사용)
runnable = prompt | llm.with_structured_output(schema=Case)

# LLM 실행
response = runnable.invoke({"text": text})

print("추출된 데이터:", response)
print(dict(response))

# 문서 객체를 페이지별로 순회하며 사례 번호 등 메타데이터를 추출하고 업데이트
pdf_docs = []
case_metadata = {}
for doc in data[1:6]:
    split_text = re.findall(split_pattern, "".join(doc.page_content))
    if split_text:

        # case id
        case_metadata['case_id'] = re.findall(r'사례\s?(\d+)\D*사건번호', split_text[0])[0]

        parts = re.split(split_pattern, "".join(doc.page_content))

        if re.search(r'주문', parts[1]):
            # Add title to metadata
            case_metadata['title'] = parts[1][:re.search(r'주문', parts[1]).span()[0]].replace('\n', '').strip()
                    
            # Update content
            doc.page_content = parts[1][re.search(r'주문', parts[1]).span()[0]:].strip()
        else:
            case_metadata['title'] = ''
            
        # Extract metadata from text
        i = 0
        while i < 10:
            try:
                response = runnable.invoke({"text": split_text[0]})
                for k, v in dict(response).items():
                    case_metadata[k] = v.replace("\n", "").replace(" ", "")
                break
            except:
                i += 1
                continue

        # Update metadata 
        doc.metadata.update(case_metadata)

        # Append to split_docs
        pdf_docs.append(doc)

    else:
        # Update metadata 
        doc.metadata.update(case_metadata) 

        # Append to split_docs
        pdf_docs.append(doc)

"""
# 총 문서 개수
print(len(pdf_docs))

pprint(pdf_docs[0].page_content)
pprint(pdf_docs[0].metadata)
"""

# Embedding
embeddings_model = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
)


# 문서를 임베딩
embeddings = embeddings_model.embed_documents(
    [
        "온라인 쇼핑몰에서 주문한 제품이 불량품으로 배송되었습니다. 이에 대한 법적 책임은 누구에게 있나요?",
        "세탁물 세탁 후 제품이 손상되었습니다. 이에 대한 피해 보상을 받을 수 있나요?",
        "항공편이 지연되어 중요한 일정을 놓쳤습니다. 이에 대한 피해 보상을 받을 수 있나요?"
    ]
)
print(len(embeddings), len(embeddings[0]))

embedded_query = embeddings_model.embed_query("에어컨 제품 불량에 대해서 보상을 받을 수 있을까요?")
embedded_query[:5]

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

# 쿼리와 문서 간의 코사인 유사도 계산
for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))


# HugoingFace Embedding 모델의 Tokenizer를 사용하여 토큰화
tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

text = "에어컨 제품 불량에 대해서 보상을 받을 수 있을까요?"
encoded = tokenizer.encode(text)
# print(len(text), len(encoded))
# print(encoded)

# Token 수를 기준으로 문서를 청크 단위로 분할
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer = tokenizer,
    chunk_size = 1024,
    chunk_overlap  = 50,
)

split_docs = text_splitter.split_documents(pdf_docs)
# print(len(split_docs))
# print(split_docs[0])

sample_text = split_docs[0].page_content
sample_encoded = tokenizer.encode(sample_text)
# print(len(sample_text), len(sample_encoded))

# print(len(tokenizer.encode(split_docs[0].page_content)))
# pprint(split_docs[0].page_content)

# 마침표 뒤에 나오는 줄바꿈 문자는 그대로 두고 나머지 줄바꿈 문자만 제거
result = re.sub(r'(?<!\.)\n', ' ', split_docs[1].page_content)

pprint(split_docs[0].metadata)

final_docs = []
for doc in split_docs:
    doc.page_content = f"### 이 사건은 '{doc.metadata['title']}'에 대한 사례입니다.\n\n" + \
    re.sub(r'(?<!\.)\n', ' ', doc.page_content)
    final_docs.append(doc)

# print(final_docs[0].page_content)


# 입력된 query를 임베딩(벡터화) 한 후, 유사도가 높은 문서 5개를 반환함 
# 단순한 유사도 기반 검색 (query와 가장 가까운 벡터 검색)
vectorstore = Chroma.from_documents(documents=final_docs, 
                                    embedding=embeddings_model, 
                                    collection_name="consumer_case_qa",
                                    persist_directory="./chroma_db")
"""
chroma_docs = vectorstore.similarity_search("세탁 후 오염에 대한 손해배상은 어떻게 이루어지나요?", k=10)
for doc in chroma_docs:
    print(str(doc.metadata["case_id"]), str(doc.metadata["page"]), doc.page_content[:200])
"""
"""
# metadata를 이용한 필터링
retriever = vectorstore.as_retriever(
    search_kwargs={
        'k': 10,
        'filter': {'case_id':'1'}
        }
)


query = "세탁 후 오염에 대한 손해배상은 어떻게 이루어지나요?"
retrieved_docs = retriever.invoke(query)

for doc in retrieved_docs:
    print(str(doc.metadata["case_id"]), str(doc.metadata["page"]), doc.page_content[:100])     
"""


# page_content를 이용한 필터링
# "세탁"이라는 단어를 포함하는 문서에서만 검색 (where_document={'$contains': '세탁'})
# 필터링된 문서 중에서 query와 유사도가 높은 문서 5개를 검색
retriever = vectorstore.as_retriever(
    search_kwargs={
        'k': 10,
        'where_document': {'$contains': '세탁'}
        }
)

query = "세탁 후 오염에 대한 손해배상은 어떻게 이루어지나요?"
retrieved_docs = retriever.invoke(query)

for doc in retrieved_docs:
    print(str(doc.metadata["case_id"]), str(doc.metadata["page"]), doc.page_content[:100])


# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt2 = ChatPromptTemplate.from_template(template)
print(prompt2)


# RAG Chain
llm2 = ChatOllama(model="qwen2", temperature=0)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

item = "세탁"
query = f"{item} 불량에 대한 손해배상은 어떻게 이루어지나요?"

retriever = vectorstore.as_retriever(
    search_kwargs={
        'k': 5,
        'where_document': {'$contains': item}
        }
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt2
    | llm2
    | StrOutputParser()
)

response = rag_chain.invoke(query)
print("응답 :", response)