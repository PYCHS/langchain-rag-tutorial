# LangChain RAG Tutorial Practice

這是一個用 Python 練習 LangChain 官方文件 RAG（Retrieval-Augmented Generation）tutorial 的練習專案。專案目前包含網頁文章、歌詞、CSV、PDF 等不同資料來源的 loader 範例，並示範如何把文件切成 chunk、建立 embedding、存進 Chroma vector store，再讓 LLM agent 透過 retrieval tool 回答問題。

## Features

- 使用 `WebBaseLoader` 抓取網頁內容
- 使用 `BeautifulSoup` 過濾 HTML 內容區塊
- 使用 `RecursiveCharacterTextSplitter` 將文件切成可檢索片段
- 使用 HuggingFace sentence-transformer 建立本地 embeddings
- 使用 Chroma 建立向量資料庫
- 使用 LangChain tool 封裝 retrieval function
- 使用 Anthropic Claude model 建立 agentic RAG 問答流程
- 額外提供 CSV 與 PDF loader 的練習範例

## Project Structure

```text
.
├── agentic_lyrics.py        # 歌詞網頁 RAG + agent 問答範例
├── example.py               # Lilian Weng agent blog RAG + agent 問答範例
├── app.py                   # 測試 .env 內 API key 是否能讀取
├── test_setup.py            # 測試 Anthropic model 呼叫
└── loader/
    ├── loaderTest.py        # CSVLoader 範例
    ├── bugsDocument.csv     # CSV 測試資料
    ├── pdfLoader.py         # PyPDFLoader 範例
    └── my_file.pdf          # PDF 測試資料
```

## Requirements

- Python 3.10+
- Anthropic API key
- macOS / Linux / Windows 皆可執行

主要套件：

```bash
pip install python-dotenv beautifulsoup4 langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-anthropic pypdf sentence-transformers
```

## Environment Setup

1. 建立並啟用虛擬環境：

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 安裝依賴：

```bash
pip install python-dotenv beautifulsoup4 langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-anthropic pypdf sentence-transformers
```

3. 建立 `.env`：

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

目前主要 RAG agent 使用的是 Anthropic；`OPENAI_API_KEY` 只是在 `app.py` 中示範讀取環境變數。

## Usage

### Run the lyrics RAG agent

```bash
python agentic_lyrics.py
```

這個 script 會：

1. 從 lyrics 網頁讀取歌詞內容
2. 將歌詞切成較小 chunks
3. 使用 HuggingFace embedding 建立向量
4. 將 chunks 加入 Chroma
5. 建立 retrieval tool
6. 呼叫 Claude agent 回答預設問題

### Run the blog RAG example

```bash
python example.py
```

這個 script 會讀取 Lilian Weng 的 agent 文章，建立 RAG 檢索流程，並詢問 task decomposition 相關問題。

### Test API key loading

```bash
python app.py
```

### Run CSV loader example

```bash
cd loader
python loaderTest.py
```

### Run PDF loader example

```bash
cd loader
python pdfLoader.py
```

## Current RAG Flow

```text
Web / CSV / PDF source
        ↓
LangChain document loader
        ↓
Text splitter
        ↓
HuggingFace embeddings
        ↓
Chroma vector store
        ↓
Retrieval tool
        ↓
Claude agent
        ↓
Answer
```

## Optimization Ideas

以下是這個專案接下來最值得優化的方向：

1. **新增 `requirements.txt` 或 `pyproject.toml`**
   目前安裝套件只能從程式 import 推回來。建議固定依賴版本，讓其他人可以穩定重現環境。

2. **避免在程式中直接寫死 query 與 URL**
   可以改成 CLI 參數，例如：

   ```bash
   python agentic_lyrics.py --url "..." --query "Who is mentioned?"
   ```

3. **把重複 RAG 流程抽成共用函式**
   `agentic_lyrics.py` 和 `example.py` 有相似的流程：load、split、embed、store、retrieve、agent。可以整理成 `rag_pipeline.py`，讓不同資料來源共用。

4. **修正 loader 範例的相對路徑**
   `loader/loaderTest.py` 和 `loader/pdfLoader.py` 目前假設執行目錄在 `loader/`。如果從 repo root 執行會找不到檔案。建議改用 `Path(__file__).parent` 建立穩定路徑。

5. **避免印出 API key**
   `app.py` 目前會直接印出 `OPENAI_API_KEY` 和 `ANTHROPIC_API_KEY`。建議只印是否讀取成功，避免 secret 外洩。

6. **設定 Chroma persistence**
   目前 Chroma 是 in-memory，程式每次執行都會重新建立向量資料庫。可以加入 `persist_directory`，讓 embedding 結果重複使用。

7. **加入錯誤處理**
   可以處理網頁讀取失敗、API key 缺失、embedding model 下載失敗、vector store 無資料等情況。

8. **增加測試**
   可以先加入簡單測試，例如確認 loader 能讀到文件、split 後 chunk 數量大於 0、retriever 能回傳結果。

## Notes

- `.env` 已經被 `.gitignore` 忽略，請不要把 API key commit 到 Git。
- 第一次使用 HuggingFace embedding model 時，`sentence-transformers/all-MiniLM-L6-v2` 可能需要下載，會花一點時間。
- 網頁 loader 依賴目標網站 HTML 結構。如果網站改版，`bs4.SoupStrainer` 的 selector 可能也需要更新。
