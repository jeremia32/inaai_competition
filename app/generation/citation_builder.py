class CitationBuilder:

    def build(self, docs):
        citations = []

        for i, doc in enumerate(docs, start=1):

            metadata = doc.metadata

            citations.append({
                "id": i,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "?"),
                "content": doc.page_content
            })

        return citations