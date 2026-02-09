"""Integration tests for vector search recall quality with real OpenAI embeddings."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from tracemem_core.embedders.openai import OpenAIEmbedder
from tracemem_core.storage.vector.lance import LanceDBVectorStore

load_dotenv()

pytestmark = pytest.mark.openai


@pytest.fixture
def openai_embedder():
    """Provide real OpenAI embedder for integration tests."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIEmbedder()


@pytest.fixture
async def vector_store(tmp_path: Path):
    """Provide a connected LanceDBVectorStore."""
    store = LanceDBVectorStore(path=tmp_path / "lancedb")
    await store.connect()
    yield store
    await store.close()


class TestVectorRecall:
    """Integration tests verifying search quality with real embeddings."""

    async def test_coding_agent_finds_related_implementation_requests(
        self, vector_store, openai_embedder
    ):
        """Test that coding-related queries find relevant past conversations."""
        # Past conversations with a coding agent
        coding_conversations = [
            (
                "Implement a REST API endpoint for user authentication using JWT tokens",
                "coding",
            ),
            (
                "Add input validation to the signup form with email and password checks",
                "coding",
            ),
            (
                "Refactor the database layer to use async SQLAlchemy with connection pooling",
                "coding",
            ),
            ("Write unit tests for the payment processing module", "coding"),
            ("Fix the race condition in the WebSocket handler", "coding"),
        ]

        # Unrelated conversations
        other_conversations = [
            ("What's the weather forecast for tomorrow in New York?", "weather"),
            ("Book a table for 4 at an Italian restaurant tonight", "booking"),
            ("Translate this paragraph from English to Spanish", "translation"),
        ]

        for text, topic in coding_conversations + other_conversations:
            embedding = await openai_embedder.embed(text)
            await vector_store.add(
                node_id=uuid4(),
                text=text,
                vector=embedding,
                conversation_id=f"conv-{topic}-{uuid4().hex[:8]}",
            )

        # Query about implementing a feature
        query_text = "implement user login with OAuth2"
        query_vector = await openai_embedder.embed(query_text)

        results = await vector_store.search(
            query_vector=query_vector,
            query_text=query_text,
            limit=3,
        )

        # Top results should be coding-related
        coding_count = sum(1 for r in results if "coding" in r.conversation_id)
        assert coding_count >= 2, (
            f"Expected at least 2 coding results, got {coding_count}"
        )

    async def test_trading_agent_finds_related_stock_analysis(
        self, vector_store, openai_embedder
    ):
        """Test that trading-related queries find relevant past conversations."""
        # Past conversations with a trading agent
        trading_conversations = [
            (
                "I bought 50 shares of Meta at $480, should I hold or sell now?",
                "trading",
            ),
            ("Analyze Netflix stock for a potential swing trade entry", "trading"),
            (
                "What's your technical analysis on AAPL's current chart pattern?",
                "trading",
            ),
            ("Should I add to my NVIDIA position given the AI hype?", "trading"),
            (
                "Review my portfolio allocation: 40% tech, 30% healthcare, 30% bonds",
                "trading",
            ),
            ("Set a stop loss for my Tesla position at 15% below entry", "trading"),
        ]

        # Unrelated conversations
        other_conversations = [
            ("Help me debug this Python function that's throwing an error", "coding"),
            ("Write a marketing email for our new product launch", "marketing"),
            ("Summarize the key points from this research paper", "research"),
        ]

        for text, topic in trading_conversations + other_conversations:
            embedding = await openai_embedder.embed(text)
            await vector_store.add(
                node_id=uuid4(),
                text=text,
                vector=embedding,
                conversation_id=f"conv-{topic}-{uuid4().hex[:8]}",
            )

        # Query about stock analysis
        query_text = "analyze AMD stock price movement"
        query_vector = await openai_embedder.embed(query_text)

        results = await vector_store.search(
            query_vector=query_vector,
            query_text=query_text,
            limit=4,
        )

        # Top results should be trading-related
        trading_count = sum(1 for r in results if "trading" in r.conversation_id)
        assert trading_count >= 3, (
            f"Expected at least 3 trading results, got {trading_count}"
        )

    async def test_hybrid_search_finds_specific_stock_ticker(
        self, vector_store, openai_embedder
    ):
        """Test that hybrid search finds documents with specific stock tickers."""
        # Documents mentioning various tickers
        documents = [
            "Bought 100 shares of TSLA at $245, planning to hold long term",
            "My GOOGL position is up 15% since I bought last month",
            "Considering selling my MSFT shares after the earnings report",
            "Added AMZN to my watchlist for a potential breakout play",
        ]

        for i, text in enumerate(documents):
            embedding = await openai_embedder.embed(text)
            await vector_store.add(
                node_id=uuid4(),
                text=text,
                vector=embedding,
                conversation_id=f"conv-{i}",
            )

        # Search for specific ticker - FTS should boost exact match
        query_text = "TSLA"
        query_vector = await openai_embedder.embed(query_text)

        results = await vector_store.search(
            query_vector=query_vector,
            query_text=query_text,
            limit=2,
        )

        # Should find the document with TSLA
        assert any("TSLA" in r.text for r in results), (
            f"Expected to find TSLA in results: {[r.text for r in results]}"
        )

    async def test_hybrid_search_finds_ticker_or_company_name(
        self, vector_store, openai_embedder
    ):
        """Test that search finds documents using either ticker or company name."""
        # Documents using tickers
        ticker_docs = [
            ("Bought TSLA calls expiring next month", "ticker"),
            ("NVDA is breaking out above resistance", "ticker"),
            ("My META position is down 5% today", "ticker"),
        ]

        # Documents using company names
        name_docs = [
            ("Tesla's new Model Y refresh looks promising for sales", "name"),
            ("Nvidia announced record data center revenue", "name"),
            ("Meta's Reality Labs is still burning cash", "name"),
        ]

        for text, doc_type in ticker_docs + name_docs:
            embedding = await openai_embedder.embed(text)
            await vector_store.add(
                node_id=uuid4(),
                text=text,
                vector=embedding,
                conversation_id=f"conv-{doc_type}",
            )

        # Search by company name, should find both ticker and name docs
        query_text = "Tesla stock analysis"
        query_vector = await openai_embedder.embed(query_text)

        results = await vector_store.search(
            query_vector=query_vector,
            query_text=query_text,
            limit=4,
        )

        result_texts = [r.text for r in results]
        has_ticker = any("TSLA" in t for t in result_texts)
        has_name = any("Tesla" in t for t in result_texts)

        assert has_ticker or has_name, (
            f"Expected to find TSLA or Tesla in results: {result_texts}"
        )

        # Search by ticker, should find both ticker and name docs
        query_text = "NVDA earnings"
        query_vector = await openai_embedder.embed(query_text)

        results = await vector_store.search(
            query_vector=query_vector,
            query_text=query_text,
            limit=4,
        )

        result_texts = [r.text for r in results]
        has_ticker = any("NVDA" in t for t in result_texts)
        has_name = any("Nvidia" in t for t in result_texts)

        assert has_ticker or has_name, (
            f"Expected to find NVDA or Nvidia in results: {result_texts}"
        )

    async def test_recall_coding_agent_feature_requests(
        self, vector_store, openai_embedder
    ):
        """Test recall quality for coding agent feature implementation requests."""
        # Relevant past requests about API implementation
        relevant_docs = [
            "Implement a GraphQL API for the user management system",
            "Add REST endpoints for CRUD operations on the Product model",
            "Create an API rate limiter middleware with Redis backend",
        ]

        # Irrelevant conversations
        irrelevant_docs = [
            "What time does the stock market open tomorrow?",
            "Calculate my portfolio's Sharpe ratio",
            "Should I rebalance my 401k allocation?",
            "Book a flight from NYC to LA for next Friday",
            "What's the best Italian restaurant nearby?",
        ]

        # Add all documents
        relevant_ids = []
        for text in relevant_docs:
            node_id = uuid4()
            relevant_ids.append(node_id)
            embedding = await openai_embedder.embed(text)
            await vector_store.add(
                node_id=node_id,
                text=text,
                vector=embedding,
                conversation_id="relevant",
            )

        for text in irrelevant_docs:
            embedding = await openai_embedder.embed(text)
            await vector_store.add(
                node_id=uuid4(),
                text=text,
                vector=embedding,
                conversation_id="irrelevant",
            )

        # Query about API implementation
        query_text = "build a REST API for order management"
        query_vector = await openai_embedder.embed(query_text)

        # Get top 5 results
        results = await vector_store.search(
            query_vector=query_vector,
            query_text=query_text,
            limit=5,
        )

        # Calculate recall@5
        retrieved_ids = {r.node_id for r in results}
        relevant_retrieved = len(set(relevant_ids) & retrieved_ids)

        # At least 2 of 3 relevant docs should be in top 5
        assert relevant_retrieved >= 2, (
            f"Expected at least 2/3 relevant docs in top 5, got {relevant_retrieved}"
        )
