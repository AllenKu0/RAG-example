from qdrant_client import QdrantClient, models
from qdrant_client.models import *
from sentence_transformers import SentenceTransformer

client = QdrantClient("http://localhost:6333")

client.create_collection(
    collection_name="Health_Tips",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

text = [
    "保持良好的手部衛生是預防感染的重要措施，經常用肥皂和清水洗手。",
    "定期接種疫苗可以預防多種傳染病，保護自己和他人的健康。",
    "每天攝取足夠的水分，成年人應該飲用約 8 杯水。",
    "每年至少進行一次全身健康檢查，有助於及早發現和治療潛在的健康問題。",
    "保持均衡的飲食，攝取足夠的蔬果、蛋白質和全穀類，減少加工食品的攝入。",
    "適量運動，每週進行至少 150 分鐘的中等強度運動，如快走或騎自行車。",
    "充足的睡眠對健康至關重要，成年人應該每晚睡 7-9 小時。",
    "避免吸菸和酗酒，這些習慣會增加多種疾病的風險，包括癌症和心臟病。",
    "定期檢查血壓，保持血壓在正常範圍內，可以減少心血管疾病的風險。",
    "保持適當的體重，減少肥胖相關的健康問題，如糖尿病和高血壓。"
]
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# embeded text
embeded_text = model.encode(text)
client.upsert(
    collection_name="Health_Tips",
    points=[
        PointStruct(id=i, vector=vector.tolist(), payload={"document": text})
        for i, (vector, text) in enumerate(zip(embeded_text, text))
    ]
)

search_text = "能不能給我一些運動的建議？"
search_results = client.search(
    collection_name="Health_Tips",
    query_vector=model.encode([search_text])[0],
    limit=1
)
print(search_results[0].payload["document"])