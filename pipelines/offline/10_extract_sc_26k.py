"""
SC Judgments 26K Extraction Pipeline

Downloads 26K SC Judgments from Kaggle, extracts text and Indian Kanoon URLs
from PDF content, and saves to JSONL.

Key Feature: Extracts doc_url from PDF pages (each page contains Indian Kanoon URL)

Usage:
    python pipelines/offline/10_extract_sc_26k.py
    python pipelines/offline/10_extract_sc_26k.py --limit 100  # Test with subset

Output:
    data/processed/sc_judgments_26k.jsonl
"""

import os
import re
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
from loguru import logger

# Configure logging
log_path = Path("logs")
log_path.mkdir(exist_ok=True)
log_file = log_path / f"sc_extract_26k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(str(log_file), rotation="50 MB", level="INFO")

# Indian Kanoon URL pattern
IK_URL_PATTERN = re.compile(r'https?://(?:www\.)?indiankanoon\.org/doc/(\d+)/?', re.IGNORECASE)
IK_URL_SIMPLE = re.compile(r'indiankanoon\.org/doc/(\d+)', re.IGNORECASE)

# Header patterns to clean
HEADER_PATTERNS = [
    r"Supreme Court of India",
    r"Page \d+ of \d+",
    r"http://JUDIS\.NIC\.IN",
    r"indiankanoon\.org/doc/\d+",  # Will be extracted, then removed from text
]


@dataclass
class ExtractedChunk:
    """Represents a single chunk from an SC judgment"""
    case_id: str
    filename: str
    doc_url: str  # Indian Kanoon URL
    case_name: str
    year: str
    chunk_idx: int
    text: str
    section: str
    pages: int


def extract_doc_url_from_pages(pages_text: List[str]) -> Tuple[Optional[str], bool]:
    """
    Extract Indian Kanoon doc_url from PDF pages.
    Returns (url, consistent) where consistent=True if all pages have same URL.
    """
    urls_found = []
    
    for page_text in pages_text:
        # Try full URL pattern first
        match = IK_URL_PATTERN.search(page_text)
        if match:
            doc_id = match.group(1)
            urls_found.append(f"https://indiankanoon.org/doc/{doc_id}/")
        else:
            # Try simple pattern
            match = IK_URL_SIMPLE.search(page_text)
            if match:
                doc_id = match.group(1)
                urls_found.append(f"https://indiankanoon.org/doc/{doc_id}/")
    
    if not urls_found:
        return None, False
    
    # Check if all URLs are the same
    unique_urls = set(urls_found)
    consistent = len(unique_urls) == 1
    
    # Return the most common URL (should be the same for all pages)
    from collections import Counter
    most_common = Counter(urls_found).most_common(1)[0][0]
    
    return most_common, consistent


def extract_case_name(text: str) -> str:
    """Extract case name from judgment text (first few lines usually contain it)"""
    # Look for patterns like "X vs Y" or "X v. Y" in first 500 chars
    first_part = text[:500]
    
    # Common patterns for case names
    patterns = [
        r'([A-Z][A-Za-z\s\.\&]+)\s+(?:vs?\.?|versus)\s+([A-Z][A-Za-z\s\.\&]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, first_part)
        if match:
            return f"{match.group(1).strip()} vs {match.group(2).strip()}"
    
    # Fallback: return first line if it looks like a title
    first_line = text.split('\n')[0].strip()[:100]
    return first_line if first_line else "Unknown"


def extract_year(text: str, filename: str) -> str:
    """Extract year from text or filename"""
    # Try filename first
    year_match = re.search(r'(19\d{2}|20\d{2})', filename)
    if year_match:
        return year_match.group(1)
    
    # Try text (look for judgment date)
    date_patterns = [
        r'(?:dated|decided).*?(19\d{2}|20\d{2})',
        r'(19\d{2}|20\d{2})\s*\)',  # Year in parentheses
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text[:1000], re.IGNORECASE)
        if match:
            return match.group(1)
    
    return "Unknown"


def clean_text(text: str) -> str:
    """Clean extracted text"""
    text = re.sub(r'\s+', ' ', text).strip()
    for pattern in HEADER_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def chunk_text(text: str, chunk_size: int = 400) -> List[Dict]:
    """Chunk text into ~400 word segments"""
    chunks = []
    paragraphs = text.split(". ")
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        current_chunk.append(para)
        current_len += len(para.split())
        
        if current_len >= chunk_size:
            chunk_text = ". ".join(current_chunk) + "."
            chunks.append({"section": "general", "text": chunk_text})
            current_chunk = []
            current_len = 0
    
    if current_chunk:
        chunk_text = ". ".join(current_chunk) + "."
        chunks.append({"section": "general", "text": chunk_text})
    
    return chunks


def extract_single_pdf(pdf_path: str) -> Optional[List[ExtractedChunk]]:
    """Extract text and URL from a single PDF (worker function)"""
    import warnings
    warnings.filterwarnings("ignore")
    
    try:
        import pdfplumber
        
        filename = os.path.basename(pdf_path)
        case_id = filename.replace(".pdf", "").replace(".PDF", "")
        
        pages_text = []
        full_text = ""
        page_count = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                    full_text += text + "\n"
        
        if not full_text.strip() or len(full_text) < 100:
            return None
        
        # Extract doc_url from pages
        doc_url, url_consistent = extract_doc_url_from_pages(pages_text)
        
        if not doc_url:
            # Fallback: construct URL placeholder
            doc_url = ""
            logger.warning(f"No URL found in {filename}")
        elif not url_consistent:
            logger.debug(f"URL mismatch across pages in {filename}, using most common")
        
        # Extract metadata
        case_name = extract_case_name(full_text)
        year = extract_year(full_text, filename)
        
        # Clean text
        cleaned = clean_text(full_text)
        if len(cleaned) < 100:
            return None
        
        # Chunk
        text_chunks = chunk_text(cleaned)
        
        chunks = []
        for idx, chunk_data in enumerate(text_chunks):
            chunks.append(ExtractedChunk(
                case_id=case_id,
                filename=filename,
                doc_url=doc_url,
                case_name=case_name,
                year=year,
                chunk_idx=idx,
                text=chunk_data["text"],
                section=chunk_data["section"],
                pages=page_count
            ))
        
        return chunks
        
    except Exception as e:
        logger.debug(f"Error processing {pdf_path}: {e}")
        return None


def run_extraction(limit: int = None):
    """Run parallel extraction"""
    import kagglehub
    
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("SC JUDGMENTS 26K EXTRACTION")
    logger.info("=" * 60)
    
    # Download dataset
    logger.info("Downloading dataset from Kaggle...")
    logger.info("Dataset: adarshsingh0903/legal-dataset-sc-judgments-india-19502024")
    cache_path = kagglehub.dataset_download("adarshsingh0903/legal-dataset-sc-judgments-india-19502024")
    logger.info(f"Dataset downloaded to: {cache_path}")
    
    # Find PDFs
    all_pdfs = glob.glob(f"{cache_path}/**/*.pdf", recursive=True)
    all_pdfs += glob.glob(f"{cache_path}/**/*.PDF", recursive=True)
    all_pdfs = list(set(all_pdfs))  # Deduplicate
    logger.info(f"Found {len(all_pdfs)} PDFs")
    
    if limit:
        all_pdfs = all_pdfs[:limit]
        logger.info(f"Limited to {limit} PDFs")
    
    # Parallel extraction
    num_workers = min(cpu_count(), 8)
    logger.info(f"Starting extraction with {num_workers} workers...")
    
    output_file = Path("data/processed/sc_judgments_26k.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    processed = 0
    failed = 0
    with_url = 0
    without_url = 0
    
    with Pool(num_workers) as pool:
        with open(output_file, "w", encoding="utf-8") as f:
            batch_size = 500
            
            for i in range(0, len(all_pdfs), batch_size):
                batch = all_pdfs[i:i+batch_size]
                results = pool.map(extract_single_pdf, batch)
                
                for chunks in results:
                    if chunks:
                        for chunk in chunks:
                            f.write(json.dumps(asdict(chunk)) + "\n")
                            total_chunks += 1
                        processed += 1
                        if chunks[0].doc_url:
                            with_url += 1
                        else:
                            without_url += 1
                    else:
                        failed += 1
                
                # Progress
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = (processed + failed) / elapsed if elapsed > 0 else 0
                eta = (len(all_pdfs) - i - len(batch)) / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {i+len(batch)}/{len(all_pdfs)} | "
                    f"OK: {processed} | Fail: {failed} | "
                    f"With URL: {with_url} | Chunks: {total_chunks} | ETA: {eta:.0f}m"
                )
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Processed: {processed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"With URL: {with_url}")
    logger.info(f"Without URL: {without_url}")
    logger.info(f"Total Chunks: {total_chunks}")
    logger.info(f"Output: {output_file} ({file_size:.1f} MB)")
    logger.info(f"Time: {elapsed:.1f} minutes")
    logger.info("=" * 60)
    logger.info("Next: Run 11_ingest_sc_26k.py to upload to Qdrant")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SC Judgments 26K Extraction")
    parser.add_argument("--limit", type=int, default=None, help="Limit PDFs to process")
    args = parser.parse_args()
    
    run_extraction(limit=args.limit)
