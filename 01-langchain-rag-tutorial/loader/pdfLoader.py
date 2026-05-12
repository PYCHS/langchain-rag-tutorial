from langchain_community.document_loaders import PyPDFLoader
# 1. Create loader for a local PDF file
loader = PyPDFLoader("my_file.pdf")

# 2. Load all pages as Document objects
documents = loader.load()

# 3. Inspect results
print(f"Loaded {len(documents)} documents")

first_doc = documents[0]
print(type(first_doc))              # Document
print(first_doc.metadata)           # often includes page number, source, etc.
print(first_doc.page_content[:500]) # first 500 chars