"""
Mock RAG (Retrieval-Augmented Generation) Module.

Simulates fetching data from a Gucci Knowledge Base.
In production, this would connect to a vector database like Pinecone or Weaviate.
"""
from typing import List, Dict, Optional


class MockGucciKnowledgeBase:
    """
    Mock knowledge base for Gucci-related information.
    In production, this would connect to a vector database.
    """

    def __init__(self):
        self.documents = self._load_mock_documents()

    def _load_mock_documents(self) -> List[Dict]:
        """Load mock knowledge base documents."""
        return [
            {
                "id": "kb_001",
                "title": "Q4 2024 Financial Results",
                "content": "Gucci Group reported revenue of €2.8B in Q4 2024, "
                          "a 12% increase year-over-year. Digital channels "
                          "now account for 35% of total sales.",
                "category": "financial",
                "relevance_score": 0.95
            },
            {
                "id": "kb_002",
                "title": "Sustainability Initiative 2025",
                "content": "Gucci's Carbon Neutral program has reduced emissions "
                          "by 40% since 2019. The brand aims for net-zero by 2030.",
                "category": "sustainability",
                "relevance_score": 0.88
            },
            {
                "id": "kb_003",
                "title": "Brand Strategy Overview",
                "content": "Gucci's brand strategy focuses on 'Eccentricity meets Elegance'. "
                          "Target demographic: 25-45 urban affluent consumers. "
                          "Key markets: China (28%), US (24%), Europe (22%).",
                "category": "strategy",
                "relevance_score": 0.92
            },
            {
                "id": "kb_004",
                "title": "Digital Transformation Roadmap",
                "content": "Investment in AI-powered personalization, virtual try-on, "
                          "and metaverse presence. NFT collection 'Gucci Garden' "
                          "generated $4M in first month.",
                "category": "digital",
                "relevance_score": 0.85
            },
            {
                "id": "kb_005",
                "title": "Supply Chain Excellence",
                "content": "100% of leather sourced from LWG-certified tanneries. "
                          "Average production time: 6-8 weeks for handbags. "
                          "Artisan training program: 2,500 craftspeople in Italy.",
                "category": "operations",
                "relevance_score": 0.78
            }
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        categories: Optional[List[str]] = None
    ) -> str:
        """
        Mock retrieval function simulating RAG.

        Args:
            query: User's question or context
            top_k: Number of documents to retrieve
            categories: Optional category filters

        Returns:
            Formatted context string for LLM prompt
        """
        query_lower = query.lower()
        scored_docs = []

        for doc in self.documents:
            if categories and doc["category"] not in categories:
                continue

            # Simple keyword-based scoring (mock similarity)
            score = 0
            for word in query_lower.split():
                if word in doc["content"].lower():
                    score += 0.1
                if word in doc["title"].lower():
                    score += 0.2

            score += doc["relevance_score"] * 0.5
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = scored_docs[:top_k]

        if not top_docs:
            return "No relevant information found in knowledge base."

        context_parts = ["[KNOWLEDGE BASE CONTEXT]"]
        for score, doc in top_docs:
            context_parts.append(f"\n--- {doc['title']} ---")
            context_parts.append(doc["content"])

        return "\n".join(context_parts)

    def get_financial_data(self) -> Dict:
        """Get mock financial data."""
        return {
            "revenue_q4_2024": 2800000000,
            "yoy_growth": 0.12,
            "digital_sales_pct": 0.35,
            "operating_margin": 0.28
        }

    def get_strategy_highlights(self) -> List[str]:
        """Get mock strategic priorities."""
        return [
            "Expand presence in Asia-Pacific region",
            "Accelerate digital transformation",
            "Strengthen sustainability credentials",
            "Enhance customer experience personalization"
        ]


# Singleton instance
_knowledge_base: Optional[MockGucciKnowledgeBase] = None


def get_knowledge_base() -> MockGucciKnowledgeBase:
    """Get or create the knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = MockGucciKnowledgeBase()
    return _knowledge_base


def mock_rag_retrieve(query: str, session_id: str = None) -> str:
    """
    Main RAG function for integration with NPC agent.

    Args:
        query: User's message or context
        session_id: Optional session for personalized retrieval

    Returns:
        Retrieved context string
    """
    kb = get_knowledge_base()
    return kb.retrieve(query)
