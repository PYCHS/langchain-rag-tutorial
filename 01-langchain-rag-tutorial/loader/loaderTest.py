from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="bugsDocument.csv")
documents = loader.load()

print(f"Loaded {len(documents)} documents")

for doc in documents:
    print("CONTENT:")
    print(doc.page_content)
    print("METADATA:")
    print(doc.metadata)
    print("-" * 40)