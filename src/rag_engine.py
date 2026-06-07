def retrieve(self, query: str) -> List[SearchResult]:
        """Retrieves top-k papers using hybrid search with index-safety and hardware-sync checks."""
        # Kiểm tra phòng thủ xem đã có dữ liệu index chưa
        if not self.papers:
            logger.warning("Retrieve called on an empty index! Please run index() first.")
            return []

        # FIX TỐI THƯỢNG CỦA VY THƯ: Encode trên GPU nhưng ép về CPU để đồng bộ với HybridSearch
        query_embedding = self.embedder.model.encode(
            query,
            convert_to_tensor=True,
            device=self.embedder.device
        ).cpu()  # Triệt tiêu hoàn toàn rủi ro Device Mismatch Bug!

        results = self.hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        search_results = []
        for rank, (idx, score) in enumerate(results):
            if idx < len(self.papers):
                search_results.append(SearchResult(
                    paper=self.papers[idx],
                    score=score,
                    rank=rank + 1
                ))
        return search_results