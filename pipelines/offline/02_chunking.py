"""
Step 2: Semantic Chunking Pipeline

This script creates semantically meaningful chunks from cleaned legal data.
For statutes (IPC/BNS): Each section = 1 chunk
For judgments: Split by section (FACTS, ARGUMENTS, JUDGMENT)

Run: python pipelines/offline/02_chunking.py

Input: data/processed/*_clean.jsonl
Output: data/processed/chunks.jsonl
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

# Configure logging
logger.add(
    "logs/chunking_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)


@dataclass
class Chunk:
    """A single chunk ready for embedding"""
    chunk_id: str
    text: str
    law_type: str
    section_num: Optional[str] = None
    case_name: Optional[str] = None
    section_type: Optional[str] = None  # FACTS, ARGUMENTS, JUDGMENT
    source_id: str = ""
    char_count: int = 0
    word_count: int = 0
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "law_type": self.law_type,
            "section_num": self.section_num,
            "case_name": self.case_name,
            "section_type": self.section_type,
            "source_id": self.source_id,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "metadata": self.metadata or {}
        }


class SemanticChunker:
    """
    Creates semantically meaningful chunks from legal documents.
    
    Strategy:
    - Statutes (IPC/BNS/BSA): 1 section = 1 chunk (already atomic)
    - Judgments: Split by rhetorical sections (FACTS, ARGUMENTS, JUDGMENT)
    - Long sections: Split at paragraph/sentence boundaries
    """
    
    # Max chunk size in words (for splitting long sections)
    MAX_CHUNK_WORDS = 500
    MIN_CHUNK_WORDS = 50
    
    # Judgment section patterns
    SECTION_PATTERNS = {
        "FACTS": [
            r"(?:the\s+)?facts\s+(?:of\s+the\s+case)?",
            r"factual\s+background",
            r"brief\s+facts",
        ],
        "ARGUMENTS": [
            r"arguments?\s+(?:advanced|of)",
            r"contentions?\s+(?:raised)?",
            r"submissions?\s+(?:made)?",
            r"(?:learned\s+)?counsel\s+(?:for|argued)",
        ],
        "JUDGMENT": [
            r"(?:our\s+)?(?:conclusion|decision|judgment)",
            r"(?:we\s+)?(?:hold|order|direct)",
            r"(?:in\s+)?(?:result|conclusion)",
            r"(?:the\s+)?appeal\s+is\s+(?:allowed|dismissed)",
        ],
        "ORDER": [
            r"(?:the\s+)?order",
            r"(?:in\s+)?(?:these\s+)?circumstances",
        ]
    }
    
    def __init__(
        self,
        processed_dir: str = "data/processed",
        max_words: int = 500
    ):
        self.processed_dir = Path(processed_dir)
        self.max_words = max_words
        
        self.stats = {
            "total_sources": 0,
            "total_chunks": 0,
            "chunks_by_type": {}
        }
    
    def chunk_statutes(self) -> Generator[Chunk, None, None]:
        """
        Process statute files (IPC, BNS, BSA).
        Each section becomes one chunk.
        """
        statute_files = [
            "ipc_clean.jsonl",
            "bns_clean.jsonl", 
            "bsa_clean.jsonl"
        ]
        
        for filename in statute_files:
            filepath = self.processed_dir / filename
            if not filepath.exists():
                logger.warning(f"Statute file not found: {filepath}")
                continue
            
            logger.info(f"Chunking statutes from: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        chunk = self._create_statute_chunk(record, line_num)
                        if chunk:
                            self.stats["total_chunks"] += 1
                            law_type = chunk.law_type
                            self.stats["chunks_by_type"][law_type] = \
                                self.stats["chunks_by_type"].get(law_type, 0) + 1
                            yield chunk
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON error in {filename} line {line_num}: {e}")
            
            self.stats["total_sources"] += 1
    
    def _create_statute_chunk(self, record: Dict, idx: int) -> Optional[Chunk]:
        """Create a chunk from a statute record"""
        
        text = record.get("text", "")
        if not text or len(text) < 20:
            return None
        
        law_type = record.get("law_type", "UNKNOWN")
        section_num = record.get("section_num", str(idx))
        chunk_id = f"{law_type}_{section_num}"
        
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            law_type=law_type,
            section_num=section_num,
            source_id=record.get("id", chunk_id),
            char_count=len(text),
            word_count=len(text.split()),
            metadata={
                "offense": record.get("offense", ""),
                "punishment": record.get("punishment", ""),
                "bailable": record.get("bailable"),
                "cognizable": record.get("cognizable"),
                "chapter": record.get("chapter", ""),
            }
        )
    
    def chunk_judgments(
        self,
        judgments_file: str = "judgments_clean.jsonl"
    ) -> Generator[Chunk, None, None]:
        """
        Process judgment files.
        Split into sections (FACTS, ARGUMENTS, JUDGMENT).
        Further split if sections are too long.
        """
        filepath = self.processed_dir / judgments_file
        if not filepath.exists():
            logger.info(f"Judgments file not found: {filepath} (skipping)")
            return
        
        logger.info(f"Chunking judgments from: {judgments_file}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    for chunk in self._chunk_judgment(record, line_num):
                        self.stats["total_chunks"] += 1
                        self.stats["chunks_by_type"]["Judgment"] = \
                            self.stats["chunks_by_type"].get("Judgment", 0) + 1
                        yield chunk
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON error in judgment line {line_num}: {e}")
        
        self.stats["total_sources"] += 1
    
    def _chunk_judgment(self, record: Dict, idx: int) -> Generator[Chunk, None, None]:
        """Split a judgment into semantic chunks"""
        
        text = record.get("text", "")
        case_id = record.get("case_id", f"case_{idx}")
        case_name = record.get("case_name", "")
        
        if not text:
            return
        
        # Try to split by sections
        sections = self._detect_sections(text)
        
        if not sections:
            # No sections detected, chunk by paragraphs
            sections = {"FULL": text}
        
        for section_type, section_text in sections.items():
            # Split long sections
            sub_chunks = self._split_long_text(section_text)
            
            for i, sub_text in enumerate(sub_chunks):
                chunk_id = f"{case_id}_{section_type}_{i+1}"
                
                yield Chunk(
                    chunk_id=chunk_id,
                    text=sub_text,
                    law_type="Judgment",
                    case_name=case_name,
                    section_type=section_type,
                    source_id=case_id,
                    char_count=len(sub_text),
                    word_count=len(sub_text.split()),
                    metadata={
                        "court": record.get("court", ""),
                        "date": record.get("date", ""),
                        "sub_chunk": i+1
                    }
                )
    
    def _detect_sections(self, text: str) -> Dict[str, str]:
        """Detect rhetorical sections in judgment text"""
        sections = {}
        text_lower = text.lower()
        
        # Find section boundaries
        boundaries = []
        
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    boundaries.append((match.start(), section_type))
        
        if not boundaries:
            return {}
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        # Extract sections
        for i, (start, section_type) in enumerate(boundaries):
            end = boundaries[i+1][0] if i+1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()
            
            if len(section_text) > 100:  # Minimum viable section
                if section_type in sections:
                    sections[section_type] += f"\n\n{section_text}"
                else:
                    sections[section_type] = section_text
        
        return sections
    
    def _split_long_text(self, text: str) -> List[str]:
        """Split long text into chunks at natural boundaries"""
        words = text.split()
        
        if len(words) <= self.max_words:
            return [text]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_word_count + para_words > self.max_words:
                # Save current chunk if substantial
                if current_word_count >= self.MIN_CHUNK_WORDS:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                
                # Handle very long paragraph
                if para_words > self.max_words:
                    sentences = self._split_into_sentences(para)
                    for sent in sentences:
                        sent_words = len(sent.split())
                        if current_word_count + sent_words > self.max_words:
                            if current_word_count >= self.MIN_CHUNK_WORDS:
                                chunks.append(' '.join(current_chunk))
                                current_chunk = []
                                current_word_count = 0
                        current_chunk.append(sent)
                        current_word_count += sent_words
                else:
                    current_chunk.append(para)
                    current_word_count += para_words
            else:
                current_chunk.append(para)
                current_word_count += para_words
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])
        
        return chunks if chunks else [text]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy)
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def run(self, output_file: str = "chunks.jsonl") -> str:
        """
        Run the chunking pipeline.
        
        Returns: Path to output file
        """
        output_path = self.processed_dir / output_file
        
        logger.info("Starting chunking pipeline...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Process statutes
            for chunk in self.chunk_statutes():
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
            
            # Process judgments (if available)
            for chunk in self.chunk_judgments():
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
        
        logger.success(f"Chunking complete → {output_path}")
        return str(output_path)
    
    def print_stats(self):
        """Print chunking statistics"""
        print("\n" + "="*50)
        print("CHUNKING SUMMARY")
        print("="*50)
        print(f"Source files processed: {self.stats['total_sources']}")
        print(f"Total chunks created:   {self.stats['total_chunks']}")
        print("\nChunks by type:")
        for law_type, count in self.stats["chunks_by_type"].items():
            print(f"  {law_type}: {count}")
        print("="*50 + "\n")


def main():
    """Main entry point for chunking pipeline"""
    print("\n" + "="*60)
    print("NEETHI APP - CHUNKING PIPELINE (Step 2)")
    print("="*60 + "\n")
    
    chunker = SemanticChunker()
    output_path = chunker.run()
    chunker.print_stats()
    
    if chunker.stats["total_chunks"] > 0:
        print(f"✅ Chunks ready: {output_path}")
        print("\nNext step: Run 04_populate_qdrant.py")
    else:
        print("⚠️  No chunks created!")
        print("Please run 01_data_cleaning.py first.")


if __name__ == "__main__":
    main()
