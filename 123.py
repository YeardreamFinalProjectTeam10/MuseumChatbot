from datasets import Dataset

from sentence_transformers import SentenceTransformer
import faiss
import time

# index = faiss.read_index("/USER/sungmin/MuseumChatbot/KorDPR_NLP/kdpr_index")
# embedder = SentenceTransformer('klue/bert-base')

# def search(query):
#     t = time.time()
#     query_embedding = embedder.encode([query])
#     k = 5
#     top_k = index.search(query_embedding, k)
#     print('total time: {}'.format(time.time() - t))
#     # 상위 5개 인덱스를 반환하려면 필요에 따라 수정하세요
#     # return top_k[1].tolist()[0]
#     return top_k

# result = search('자유학기제가 뭐야?')
# print(result)



ds = Dataset.from_file("/USER/sungmin/MuseumChatbot/KorDPR_NLP/kdpr_train/data-00000-of-00001.arrow")
print(ds)