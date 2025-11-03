import os
import re
import shutil

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# Load environment variables from the .env file
load_dotenv()


def extract_gri_standard_number(filename: str) -> str | None:
    """파일명에서 GRI 표준 번호를 추출합니다.
    
    예: "GRI 101_ Biodiversity 2024 - English.pdf" -> "GRI 101"
        "GRI 2_ General Disclosures 2021.pdf" -> "GRI 2"
        "GRI 11_ Oil and Gas Sector 2021.pdf" -> "GRI 11"
    """
    pattern = r"GRI\s+(\d+[A-Z]?)\s*[_:]"
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return f"GRI {match.group(1)}"
    return None


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
    recursive: bool = True,
    extra_metadata: dict[str, str] | None = None,
    collection_name: str | None = None,
):
    """벡터 데이터베이스를 생성합니다.
    
    Args:
        folder_path: 문서가 있는 폴더 경로
        db_name: 벡터 DB 저장 경로 (기본값: "./chroma_db")
        delete_chroma_db: 기존 DB 삭제 여부
        chunk_size: 청크 크기
        overlap: 청크 겹침 크기
        recursive: 하위 폴더 재귀 탐색 여부
        extra_metadata: 추가할 메타데이터 딕셔너리 (모든 청크에 적용)
        collection_name: 컬렉션 이름 (None이면 기본 컬렉션 사용)
    """
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    # Chroma는 persist_directory가 있으면 자동으로 기존 컬렉션을 로드합니다
    # collection_name을 지정하면 해당 컬렉션을 사용하고, 없으면 기본 컬렉션 사용
    # 같은 DB 내에서 다른 컬렉션으로 문서를 분리 저장 가능
    
    if not delete_chroma_db and os.path.exists(db_name):
        # 기존 DB가 있으면 지정된 컬렉션에 추가 (또는 기본 컬렉션)
        if collection_name:
            print(f"Adding documents to existing collection '{collection_name}' in database at {db_name}")
        else:
            print(f"Adding documents to existing database at {db_name}")
        chroma = Chroma(
            embedding_function=embeddings,
            persist_directory=db_name,
            collection_name=collection_name,
        )
    else:
        # 새 DB 생성 (또는 기존 DB가 없을 때)
        if collection_name:
            print(f"Creating new collection '{collection_name}' in database at {db_name}")
        else:
            print(f"Creating new database at {db_name}")
        chroma = Chroma(
            embedding_function=embeddings,
            persist_directory=db_name,
            collection_name=collection_name,
        )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Get all files to process
    files_to_process = []
    if recursive:
        # 재귀적으로 모든 하위 폴더 탐색
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith((".pdf", ".docx")):
                    file_path = os.path.join(root, filename)
                    # 상대 경로를 사용하여 카테고리 정보 추출
                    rel_path = os.path.relpath(file_path, folder_path)
                    # 카테고리 정보 추출 (첫 번째 하위 폴더명)
                    category = rel_path.split(os.sep)[0] if os.sep in rel_path else None
                    files_to_process.append((file_path, filename, category))
    else:
        # 단일 폴더만 처리
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith((".pdf", ".docx")):
                files_to_process.append((file_path, filename, None))

    # Process each file
    total_files = len(files_to_process)
    print(f"Found {total_files} files to process.")

    for idx, (file_path, filename, category) in enumerate(files_to_process, 1):
        print(f"\n[{idx}/{total_files}] Processing: {filename}")

        # Load document based on file extension
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue

        # Load and split document into chunks
        try:
            document = loader.load()
            chunks = text_splitter.split_documents(document)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Extract GRI standard number from filename
        gri_standard = extract_gri_standard_number(filename)

        # Add metadata to each chunk
        for chunk in chunks:
            # 기존 메타데이터 유지
            if not chunk.metadata:
                chunk.metadata = {}
            
            # 카테고리 정보 추가
            if category:
                chunk.metadata["category"] = category
            
            # GRI 표준 번호 추가
            if gri_standard:
                chunk.metadata["gri_standard"] = gri_standard
            
            # 원본 파일명 명시적 추가
            chunk.metadata["filename"] = filename
            
            # 추가 메타데이터가 있으면 모든 청크에 적용
            if extra_metadata:
                chunk.metadata.update(extra_metadata)

        # Add chunks to Chroma vector store
        try:
            # 지정된 컬렉션에 문서 추가
            collection_info = f"collection: {collection_name}" if collection_name else "default collection"
            chroma.add_documents(chunks)
            print(f"  ✓ Added {len(chunks)} chunks to database ({collection_info})")
        except Exception as e:
            print(f"  ✗ Error adding chunks: {e}")
            raise  # 에러를 다시 발생시켜서 호출자가 처리할 수 있도록 함
    
    # Chroma는 자동으로 persist되지만, 명시적으로 persist하여 저장 보장
    # persist_directory를 사용하면 자동으로 저장되므로 추가 작업 불필요
    
    print(f"\n✓ Vector database created and saved in {db_name}.")
    print(f"  Total files processed: {total_files}")
    if collection_name:
        print(f"  Collection: {collection_name}")
    return chroma


if __name__ == "__main__":
    # Path to the folder containing the documents
    folder_path = "./data"
    db_name = "./chroma_db"

    # Create the Chroma database
    chroma = create_chroma_db(
        folder_path=folder_path,
        db_name=db_name,
        recursive=True,
    )

    # Create retriever from the Chroma database
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # Perform a similarity search
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
