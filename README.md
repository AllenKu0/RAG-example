# RAG-example
## Qdrant 安裝與執行(Docker)
```
// pull image
docker pull qdrant/qdrant
// run qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```    
## Python 環境安裝
```
pip install qdrant-client langchain-community sentence-transformers gradio==4.32.1
```

## 啟動
```
python app.py
```
