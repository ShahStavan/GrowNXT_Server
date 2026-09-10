"""Comprehensive verification test suite for multimodal PDF element chunker.

Tests compliance with the CocoIndex PDF Elements architecture and LlamaIndex
RecursiveTextSplitter integration:

1. Text splitting using LlamaIndex SentenceSplitter with boundary preservation.
2. Table chunking with header repetition on large partitioned financial tables.
3. Figure/chart chunking with pixel filtering, classification, and image path tracking.
4. Heading breadcrumb propagation and page coordinates.
5. Deterministic fingerprinting and JSON serialization round-trips.
6. Real filing extraction processing (Annual Report, Presentation, Transcript).

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ingestion.chunker import (
    ELEMENT_FIGURE,
    ELEMENT_TABLE,
    ELEMENT_TEXT,
    Chunk,
    ChunkSet,
    chunk_document,
)
from ingestion.documents.content import (
    KIND_FIGURE,
    KIND_HEADING,
    KIND_TABLE,
    KIND_TEXT,
    Block,
    ExtractedDocument,
    Figure,
    Table,
)


class TestSentenceSplitter(unittest.TestCase):
    """Verifies recursive text splitting logic."""

    def test_small_text_stays_single_chunk(self) -> None:
        doc = ExtractedDocument(
            doc_id="test_doc",
            doc_type="concall_presentation",
            ticker="TEST",
            label="Test Presentation",
            blocks=[
                Block(kind=KIND_HEADING, text="Overview", path=["Overview"], page=1),
                Block(
                    kind=KIND_TEXT,
                    text="Short business update paragraph.",
                    path=["Overview"],
                    page=1,
                ),
            ],
        )
        chunk_set = chunk_document(doc, chunk_size=500, chunk_overlap=50)
        self.assertEqual(chunk_set.n_chunks, 1)
        chunk = chunk_set.chunks[0]
        self.assertEqual(chunk.element_type, ELEMENT_TEXT)
        self.assertEqual(chunk.page_start, 1)
        self.assertEqual(chunk.path, ["Overview"])
        self.assertIn("Short business update paragraph.", chunk.text)
        self.assertIn("Section: Overview", chunk.embed_text)

    def test_long_prose_is_recursively_split(self) -> None:
        long_paragraph = (
            "Operating margins improved across all three business units during the fiscal period. "
            * 25
        )
        doc = ExtractedDocument(
            doc_id="test_long",
            doc_type="annual_report",
            ticker="TEST",
            label="Annual Report",
            blocks=[
                Block(kind=KIND_HEADING, text="MD&A", path=["MD&A"], page=5),
                Block(kind=KIND_TEXT, text=long_paragraph, path=["MD&A"], page=5),
            ],
        )
        chunk_set = chunk_document(doc, chunk_size=300, chunk_overlap=50)
        self.assertGreater(chunk_set.n_chunks, 1)
        for chunk in chunk_set.chunks:
            self.assertEqual(chunk.element_type, ELEMENT_TEXT)
            self.assertEqual(chunk.page_start, 5)
            self.assertGreater(len(chunk.text), 0)
            self.assertLess(len(chunk.text), 2000)


class TestTableChunking(unittest.TestCase):
    """Verifies structured table preservation and large table partitioning."""

    def test_small_table_stays_intact(self) -> None:
        table = Table(
            rows=[
                ["Segment", "Revenue (INR Cr)", "EBITDA Margin"],
                ["IT Services", "22,500", "16.4%"],
                ["Consulting", "4,200", "14.1%"],
            ],
            header_rows=1,
            caption="Segment Performance",
            page=12,
        )
        doc = ExtractedDocument(
            doc_id="test_tbl_small",
            doc_type="annual_report",
            ticker="TEST",
            label="Annual Report",
            blocks=[
                Block(
                    kind=KIND_HEADING,
                    text="Segment Review",
                    path=["Segment Review"],
                    page=12,
                ),
                Block(
                    kind=KIND_TABLE,
                    text=table.to_markdown(),
                    path=["Segment Review"],
                    table=table,
                    page=12,
                ),
            ],
        )
        chunk_set = chunk_document(doc)
        self.assertEqual(chunk_set.n_chunks, 1)
        chunk = chunk_set.chunks[0]
        self.assertEqual(chunk.element_type, ELEMENT_TABLE)
        self.assertEqual(chunk.page_start, 12)
        self.assertIn("IT Services", chunk.text)
        self.assertIn("Consulting", chunk.text)
        self.assertEqual(chunk.metadata["n_rows"], 3)
        self.assertEqual(chunk.metadata["header_rows"], 1)

    def test_large_table_is_partitioned_with_headers_repeated(self) -> None:
        header = [["Line Item", "FY26", "FY25"]]
        body = [
            [f"Schedule Item {i}", f"{1000 + i}", f"{900 + i}"] for i in range(1, 60)
        ]
        table = Table(
            rows=header + body,
            header_rows=1,
            caption="Consolidated Financial Schedules",
            page=20,
        )
        doc = ExtractedDocument(
            doc_id="test_tbl_large",
            doc_type="annual_report",
            ticker="TEST",
            label="Annual Report",
            blocks=[
                Block(
                    kind=KIND_HEADING, text="Financials", path=["Financials"], page=20
                ),
                Block(
                    kind=KIND_TABLE,
                    text=table.to_markdown(),
                    path=["Financials"],
                    table=table,
                    page=20,
                ),
            ],
        )
        chunk_set = chunk_document(doc)
        self.assertGreater(chunk_set.n_chunks, 1)
        for chunk in chunk_set.chunks:
            self.assertEqual(chunk.element_type, ELEMENT_TABLE)
            self.assertIn("Line Item", chunk.text)
            self.assertIn("FY26", chunk.text)
            self.assertIn("Table Part", chunk.embed_text)


class TestFigureChunking(unittest.TestCase):
    """Verifies visual element and figure metadata extraction."""

    def test_meaningful_chart_creates_figure_chunk(self) -> None:
        fig = Figure(
            path="figures/test_fig/p0004-01.png",
            page=4,
            kind="bar_chart",
            width=800,
            height=400,
            caption="Quarterly Revenue Trend",
        )
        doc = ExtractedDocument(
            doc_id="test_fig",
            doc_type="concall_presentation",
            ticker="TEST",
            label="Investor Presentation",
            blocks=[
                Block(
                    kind=KIND_HEADING,
                    text="Revenue Growth",
                    path=["Revenue Growth"],
                    page=4,
                ),
                Block(
                    kind=KIND_FIGURE,
                    text="",
                    path=["Revenue Growth"],
                    figure=fig,
                    page=4,
                ),
            ],
        )
        chunk_set = chunk_document(doc)
        self.assertEqual(chunk_set.n_chunks, 1)
        chunk = chunk_set.chunks[0]
        self.assertEqual(chunk.element_type, ELEMENT_FIGURE)
        self.assertEqual(chunk.page_start, 4)
        self.assertEqual(chunk.figure_path, "figures/test_fig/p0004-01.png")
        self.assertIn("figures/test_fig/p0004-01.png", chunk.text)
        self.assertEqual(chunk.metadata["kind"], "bar_chart")
        self.assertEqual(chunk.metadata["width"], 800)
        self.assertEqual(chunk.metadata["height"], 400)

    def test_tiny_decorative_furniture_is_filtered(self) -> None:
        tiny_icon = Figure(
            path="figures/test_fig/icon.png",
            page=1,
            kind="icon",
            width=24,
            height=24,
            caption="",
        )
        doc = ExtractedDocument(
            doc_id="test_filter",
            doc_type="concall_presentation",
            ticker="TEST",
            label="Presentation",
            blocks=[
                Block(kind=KIND_FIGURE, text="", path=[], figure=tiny_icon, page=1),
            ],
        )
        chunk_set = chunk_document(doc, min_figure_pixels=48)
        self.assertEqual(chunk_set.n_chunks, 0)


class TestChunkSerialization(unittest.TestCase):
    """Verifies chunk serialization and persistence."""

    def test_chunk_and_chunkset_roundtrip(self) -> None:
        chunk = Chunk(
            chunk_id="test_p0001_txt_001",
            doc_id="test_doc",
            ticker="TEST",
            doc_type="annual_report",
            label="FY26 Report",
            element_type=ELEMENT_TEXT,
            page_start=1,
            page_end=1,
            section="Executive Summary",
            path=["Executive Summary", "Strategic Highlights"],
            text="Revenue increased 12% YoY.",
            embed_text="[TEST] Revenue increased 12% YoY.",
            metadata={"char_count": 26},
        )
        chunk_set = ChunkSet(
            doc_id="test_doc",
            ticker="TEST",
            doc_type="annual_report",
            label="FY26 Report",
            fingerprint="abc123hash",
            params={"chunk_size": 800},
            chunks=[chunk],
        )

        serialized = chunk_set.to_dict()
        reconstructed = ChunkSet.from_dict(serialized)

        self.assertEqual(reconstructed.doc_id, chunk_set.doc_id)
        self.assertEqual(reconstructed.fingerprint, chunk_set.fingerprint)
        self.assertEqual(reconstructed.n_chunks, 1)
        r_chunk = reconstructed.chunks[0]
        self.assertEqual(r_chunk.chunk_id, chunk.chunk_id)
        self.assertEqual(r_chunk.element_type, chunk.element_type)
        self.assertEqual(r_chunk.path, chunk.path)

    def test_llamaindex_node_conversion(self) -> None:
        chunk_txt = Chunk(
            chunk_id="test_txt_001",
            doc_id="test_doc",
            ticker="TEST",
            element_type=ELEMENT_TEXT,
            page_start=2,
            page_end=2,
            section="Overview",
            text="Prose content for LlamaIndex node.",
        )
        chunk_fig = Chunk(
            chunk_id="test_fig_001",
            doc_id="test_doc",
            ticker="TEST",
            element_type=ELEMENT_FIGURE,
            page_start=4,
            page_end=4,
            figure_path="figures/test_doc/p0004-01.png",
            text="Figure markdown description.",
        )
        chunk_set = ChunkSet(
            doc_id="test_doc",
            ticker="TEST",
            chunks=[chunk_txt, chunk_fig],
        )

        nodes = chunk_set.to_nodes()
        self.assertEqual(len(nodes), 2)
        from llama_index.core.schema import ImageNode, TextNode

        self.assertIsInstance(nodes[0], TextNode)
        self.assertIsInstance(nodes[1], ImageNode)
        self.assertEqual(nodes[1].image_path, "figures/test_doc/p0004-01.png")

        reconstructed_chunk = Chunk.from_node(nodes[0])
        self.assertEqual(reconstructed_chunk.chunk_id, "test_txt_001")
        self.assertEqual(reconstructed_chunk.text, "Prose content for LlamaIndex node.")

        from llama_index.core.schema import NodeRelationship

        self.assertIn(NodeRelationship.NEXT, nodes[0].relationships)
        self.assertEqual(
            nodes[0].relationships[NodeRelationship.NEXT].node_id, nodes[1].node_id
        )
        self.assertIn(NodeRelationship.PREVIOUS, nodes[1].relationships)
        self.assertEqual(
            nodes[1].relationships[NodeRelationship.PREVIOUS].node_id, nodes[0].node_id
        )

        reconstructed_set = ChunkSet.from_nodes(nodes, doc_id="test_doc", ticker="TEST")
        self.assertEqual(reconstructed_set.n_chunks, 2)


class TestRealExtractions(unittest.TestCase):
    """Verifies chunking against real extracted corporate filings."""

    def setUp(self) -> None:
        self.extracted_dir = Path("output/WIPRO/extracted")

    def test_presentation_chunking(self) -> None:
        path = self.extracted_dir / "presentation_2026_01.json"
        if not path.exists():
            self.skipTest("Extracted presentation not available")
        doc = ExtractedDocument.from_dict(json.loads(path.read_text("utf-8")))
        chunk_set = chunk_document(doc)
        counts = chunk_set.element_counts()
        self.assertGreater(chunk_set.n_chunks, 20)
        self.assertGreater(counts.get(ELEMENT_FIGURE, 0), 10)
        self.assertGreater(counts.get(ELEMENT_TEXT, 0), 10)

    def test_annual_report_chunking(self) -> None:
        path = self.extracted_dir / "annual_report_FY2026_financials.json"
        if not path.exists():
            self.skipTest("Extracted annual report not available")
        doc = ExtractedDocument.from_dict(json.loads(path.read_text("utf-8")))
        chunk_set = chunk_document(doc)
        counts = chunk_set.element_counts()
        self.assertGreater(chunk_set.n_chunks, 30)
        self.assertGreater(counts.get(ELEMENT_TABLE, 0), 5)
        self.assertGreater(counts.get(ELEMENT_TEXT, 0), 30)

    def test_transcript_chunking(self) -> None:
        path = self.extracted_dir / "transcript_2026_07.json"
        if not path.exists():
            self.skipTest("Extracted transcript not available")
        doc = ExtractedDocument.from_dict(json.loads(path.read_text("utf-8")))
        chunk_set = chunk_document(doc)
        self.assertGreater(chunk_set.n_chunks, 20)


if __name__ == "__main__":
    unittest.main()
