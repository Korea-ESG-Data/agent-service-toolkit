import math
import os
import re
from pathlib import Path

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings

# 프로젝트 루트 디렉토리 (tools.py는 src/agents/에 있으므로 상위 2단계)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # Create the embedding function for our project description database
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


def load_esg_standards_db():
    """GRI 표준 벡터 데이터베이스를 로드합니다.
    
    같은 DB 내의 두 컬렉션을 모두 반환:
    - esg_standards_collection: GRI 표준 문서
    - uploaded_reports_collection: 업로드된 보고서
    
    같은 DB 경로와 컬렉션 이름을 사용하여 일관성 보장.
    """
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # 프로젝트 루트 기준 절대 경로 사용 (업로드와 동일한 경로)
    db_name = PROJECT_ROOT / "chroma_db_esg_standards"
    db_name_str = str(db_name.resolve())
    
    # 두 컬렉션을 모두 로드 (업로드와 동일한 컬렉션 이름 사용)
    standards_collection = "esg_standards_collection"
    reports_collection = "uploaded_reports_collection"
    
    # DB 경로와 컬렉션 이름을 확인하여 로깅 (디버깅용)
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Loading collections from DB: {db_name_str}")
    logger.debug(f"Standards collection: {standards_collection}, Reports collection: {reports_collection}")
    
    try:
        standards_db = Chroma(
            persist_directory=db_name_str,
            embedding_function=embeddings,
            collection_name=standards_collection,
        )
    except Exception as e:
        logger.warning(f"Could not load standards collection: {e}")
        standards_db = None
    
    try:
        reports_db = Chroma(
            persist_directory=db_name_str,
            embedding_function=embeddings,
            collection_name=reports_collection,
        )
    except Exception as e:
        logger.warning(f"Could not load reports collection: {e}")
        reports_db = None
    
    return {
        "standards": standards_db,
        "reports": reports_db,
    }


def format_esg_contexts(docs):
    """ESG 표준 문서 검색 결과를 포맷팅합니다.
    
    메타데이터(카테고리, GRI 표준 번호, source_type 등)를 포함하여 반환합니다.
    업로드된 보고서와 GRI 표준 문서를 구분하여 표시합니다.
    """
    formatted_results = []
    for doc in docs:
        metadata_parts = []
        
        # source_type 구분 표시
        source_type = doc.metadata.get("source_type")
        if source_type == "uploaded_report":
            metadata_parts.append("[업로드된 보고서]")
        
        if doc.metadata.get("gri_standard"):
            metadata_parts.append(f"표준: {doc.metadata['gri_standard']}")
        
        if doc.metadata.get("category"):
            metadata_parts.append(f"카테고리: {doc.metadata['category']}")
        
        if doc.metadata.get("filename"):
            metadata_parts.append(f"파일: {doc.metadata['filename']}")
        
        if doc.metadata.get("upload_timestamp"):
            metadata_parts.append(f"업로드: {doc.metadata['upload_timestamp']}")
        
        metadata_str = " | ".join(metadata_parts) if metadata_parts else ""
        
        content = doc.page_content
        if metadata_str:
            formatted_results.append(f"[{metadata_str}]\n{content}")
        else:
            formatted_results.append(content)
    
    return "\n\n---\n\n".join(formatted_results)


def esg_standards_search_func(query: str) -> str:
    """Searches the unified ESG Standards database for GRI Standards documents and uploaded sustainability reports.
    
    This is the ONLY search tool available. Use this tool for ALL searches including:
    - GRI Standards questions (e.g., "What is GRI 305?", "How should I report emissions?")
    - Uploaded sustainability report questions (e.g., "What is our company's carbon emissions?", "What are our ESG goals?")
    - Company-specific information from uploaded reports
    - Any information related to GRI Standards or sustainability reporting
    
    The search results include metadata indicating whether a document is:
    - An official GRI Standard (includes "표준: GRI XXX" and "카테고리: ...")
    - An uploaded report (marked with "[업로드된 보고서]" tag)
    
    IMPORTANT: This is the ONLY search tool. Do NOT use report_search or any other search tools - they do not exist.
    
    Args:
        query: The question or keywords to search for
        
    Returns:
        Retrieved document content with metadata from GRI Standards and uploaded reports
    """
    # 두 컬렉션을 모두 로드
    chroma_dbs = load_esg_standards_db()
    
    # 각 컬렉션에서 검색
    all_documents = []
    error_messages = []
    
    # GRI 표준 컬렉션 검색
    if chroma_dbs["standards"] is not None:
        try:
            standards_retriever = chroma_dbs["standards"].as_retriever(search_kwargs={"k": 3})
            standards_docs = standards_retriever.invoke(query)
            all_documents.extend(standards_docs)
        except Exception as e:
            # 컬렉션 검색 오류
            error_messages.append(f"GRI 표준 컬렉션 검색 오류: {str(e)}")
    else:
        error_messages.append("GRI 표준 컬렉션을 로드할 수 없습니다.")
    
    # 업로드된 보고서 컬렉션 검색
    if chroma_dbs["reports"] is not None:
        try:
            reports_retriever = chroma_dbs["reports"].as_retriever(search_kwargs={"k": 3})
            reports_docs = reports_retriever.invoke(query)
            all_documents.extend(reports_docs)
        except Exception as e:
            # 컬렉션 검색 오류
            error_messages.append(f"업로드된 보고서 컬렉션 검색 오류: {str(e)}")
    else:
        error_messages.append("업로드된 보고서 컬렉션을 로드할 수 없습니다. (보고서가 아직 업로드되지 않았을 수 있습니다.)")
    
    # 검색 결과가 없으면
    if not all_documents:
        error_info = "\n".join(error_messages) if error_messages else ""
        return (
            "검색 결과가 없습니다. GRI 표준 문서나 업로드된 보고서에서 관련 정보를 찾을 수 없습니다."
            + (f"\n[디버그 정보: {error_info}]" if error_info else "")
        )

    # Format the documents with metadata
    context_str = format_esg_contexts(all_documents)

    return context_str


esg_standards_search: BaseTool = tool(esg_standards_search_func)
esg_standards_search.name = "ESG_Standards_Search"
