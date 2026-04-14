import os
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(id="song-body")
loader = WebBaseLoader(
    web_paths=("https://lyricstranslate.com/en/various-artists-kim-jung-goon-lyrics",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[:1000])
# print(f"Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content[:500])