from rank_bm25 import BM25Okapi
import numpy as np

class CustomBM25(BM25Okapi):
    def get_top_n(self, query, ctxids, neg_num):
        scores = self.get_scores(query)
        top_docs_indices = np.argsort(scores)[::-1][:neg_num]
        top_ctx_idx = [ctxids[index] for index in top_docs_indices]
        return top_ctx_idx