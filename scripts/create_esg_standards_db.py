"""GRI 표준 문서를 벡터화하여 Chroma DB를 생성하는 스크립트"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_chroma_db import create_chroma_db

if __name__ == "__main__":
    # GRI 표준 문서가 있는 디렉토리 경로
    gri_standards_path = os.path.expanduser("~/work/gri-standards")
    
    # 벡터 DB 저장 경로
    db_name = "./chroma_db_esg_standards"
    
    # 경로 확인
    if not os.path.exists(gri_standards_path):
        print(f"Error: GRI standards directory not found at {gri_standards_path}")
        print("Please ensure the gri-standards directory exists.")
        exit(1)
    
    print(f"Creating ESG Standards vector database...")
    print(f"Source: {gri_standards_path}")
    print(f"Destination: {db_name}")
    print("-" * 60)
    
    # Create the Chroma database with esg_standards_collection
    # 업로드된 보고서는 uploaded_reports_collection을 사용하므로 분리됨
    chroma = create_chroma_db(
        folder_path=gri_standards_path,
        db_name=db_name,
        delete_chroma_db=True,
        recursive=True,
        chunk_size=2000,
        overlap=500,
        collection_name="esg_standards_collection",  # GRI 표준 전용 컬렉션
    )
    
    print("-" * 60)
    print(f"✓ Vector database created successfully at {db_name}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    retriever = chroma.as_retriever(search_kwargs={"k": 3})
    test_query = "What is biodiversity reporting?"
    similar_docs = retriever.invoke(test_query)
    
    print(f"\nTest query: '{test_query}'")
    print(f"Found {len(similar_docs)} documents:")
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n[{i}] Metadata: {doc.metadata}")
        print(f"    Content preview: {doc.page_content[:150]}...")

