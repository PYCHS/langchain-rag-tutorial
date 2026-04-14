import bs4
import os
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

from langchain_text_splitters import RecursiveCharacterTextSplitter
print(f"Original numbers of docs: {len(docs)}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# 3. Local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector store
vector_store = Chroma(
    collection_name="lyrics_collection",
    embedding_function=embeddings,
)

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
print(len(document_ids))