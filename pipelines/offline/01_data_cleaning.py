"""
Step 1: Data Cleaning Pipeline

This script cleans and standardizes raw legal datasets:
- IPC (Indian Penal Code) sections
- BNS (Bharatiya Nyaya Sanhita) sections  
- BSA (Bharatiya Sakshya Adhiniyam) sections

Run: python pipelines/offline/01_data_cleaning.py

Output: data/processed/*.jsonl
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime

# Configure logging
logger.add(
    "logs/data_cleaning_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)


class LegalDataCleaner:
    """
    Cleans and standardizes legal datasets for RAG ingestion.
    """
    
    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed"
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "total_records": 0,
            "cleaned_records": 0,
            "skipped_records": 0,
            "errors": []
        }
    
    def clean_ipc_dataset(self, filename: str = "ipc_sections.csv") -> str:
        """
        Clean the IPC sections dataset from Kaggle.
        
        Expected columns: Section, Description, Offense, Punishment
        Output: Standardized JSONL with consistent fields
        """
        logger.info("Starting IPC dataset cleaning...")
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            logger.info(f"Please download from Kaggle and place in {self.raw_dir}")
            return ""
        
        try:
            # Read CSV with flexible encoding
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Loaded {len(df)} IPC records")
            self.stats["total_records"] += len(df)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            # Try alternative encodings
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
            except:
                self.stats["errors"].append(f"Failed to read {filename}")
                return ""
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Map to standard schema
        column_mapping = {
            'section': 'section_num',
            'section_no': 'section_num',
            'sec': 'section_num',
            'description': 'description',
            'offense': 'offense',
            'offence': 'offense',
            'punishment': 'punishment',
            'penalty': 'punishment',
            'cognizable': 'cognizable',
            'bailable': 'bailable',
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Clean and process each record
        cleaned_records = []
        for idx, row in df.iterrows():
            try:
                record = self._clean_ipc_record(row, idx)
                if record:
                    cleaned_records.append(record)
                    self.stats["cleaned_records"] += 1
                else:
                    self.stats["skipped_records"] += 1
            except Exception as e:
                logger.warning(f"Error cleaning row {idx}: {e}")
                self.stats["skipped_records"] += 1
        
        # Save to JSONL
        output_path = self.processed_dir / "ipc_clean.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.success(f"Cleaned {len(cleaned_records)} IPC sections → {output_path}")
        return str(output_path)
    
    def _clean_ipc_record(self, row: pd.Series, idx: int) -> Optional[Dict]:
        """Clean a single IPC record"""
        
        # Extract section number
        section_num = str(row.get('section_num', row.get('section', ''))).strip()
        if not section_num or section_num.lower() == 'nan':
            return None
        
        # Normalize section number (e.g., "302" → "IPC_302")
        section_num = re.sub(r'[^\dA-Za-z]', '', section_num)
        section_id = f"IPC_{section_num}"
        
        # Get description
        description = str(row.get('description', '')).strip()
        if not description or description.lower() == 'nan':
            description = "Description not available"
        
        # Get offense type
        offense = str(row.get('offense', '')).strip()
        if offense.lower() == 'nan':
            offense = ""
        
        # Get punishment
        punishment = str(row.get('punishment', '')).strip()
        if punishment.lower() == 'nan':
            punishment = ""
        
        # Get bailable status
        bailable = str(row.get('bailable', '')).strip().lower()
        bailable = True if bailable in ['yes', 'true', 'bailable', '1'] else \
                   False if bailable in ['no', 'false', 'non-bailable', '0'] else None
        
        # Get cognizable status  
        cognizable = str(row.get('cognizable', '')).strip().lower()
        cognizable = True if cognizable in ['yes', 'true', 'cognizable', '1'] else \
                     False if cognizable in ['no', 'false', 'non-cognizable', '0'] else None
        
        # Create combined text for embedding
        text_parts = [f"IPC Section {section_num}: {description}"]
        if offense:
            text_parts.append(f"Offense: {offense}")
        if punishment:
            text_parts.append(f"Punishment: {punishment}")
        
        combined_text = ". ".join(text_parts)
        
        return {
            "id": section_id,
            "section_num": section_num,
            "law_type": "IPC",
            "description": description,
            "offense": offense,
            "punishment": punishment,
            "bailable": bailable,
            "cognizable": cognizable,
            "text": combined_text,
            "char_count": len(combined_text),
            "processed_at": datetime.now().isoformat()
        }
    
    def clean_bns_dataset(self, filename: str = "bns_sections.csv") -> str:
        """
        Clean the BNS (Bharatiya Nyaya Sanhita) dataset.
        
        Expected columns: Chapter, Section, Section_name, Description
        """
        logger.info("Starting BNS dataset cleaning...")
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return ""
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Loaded {len(df)} BNS records")
            self.stats["total_records"] += len(df)
        except Exception as e:
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
            except:
                self.stats["errors"].append(f"Failed to read {filename}")
                return ""
        
        # Standardize columns
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        cleaned_records = []
        for idx, row in df.iterrows():
            try:
                record = self._clean_bns_record(row, idx)
                if record:
                    cleaned_records.append(record)
                    self.stats["cleaned_records"] += 1
                else:
                    self.stats["skipped_records"] += 1
            except Exception as e:
                logger.warning(f"Error cleaning BNS row {idx}: {e}")
                self.stats["skipped_records"] += 1
        
        output_path = self.processed_dir / "bns_clean.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.success(f"Cleaned {len(cleaned_records)} BNS sections → {output_path}")
        return str(output_path)
    
    def _clean_bns_record(self, row: pd.Series, idx: int) -> Optional[Dict]:
        """Clean a single BNS record"""
        
        section_num = str(row.get('section', '')).strip()
        if not section_num or section_num.lower() == 'nan':
            return None
        
        section_num = re.sub(r'[^\dA-Za-z]', '', section_num)
        section_id = f"BNS_{section_num}"
        
        section_name = str(row.get('section_name', '')).strip()
        if section_name.lower() == 'nan':
            section_name = ""
        
        description = str(row.get('description', '')).strip()
        if not description or description.lower() == 'nan':
            description = section_name if section_name else "Description not available"
        
        chapter = str(row.get('chapter', '')).strip()
        chapter_name = str(row.get('chapter_name', '')).strip()
        
        # Combined text
        text_parts = [f"BNS Section {section_num}"]
        if section_name:
            text_parts[0] += f" ({section_name})"
        text_parts[0] += f": {description}"
        if chapter and chapter_name:
            text_parts.append(f"Chapter {chapter}: {chapter_name}")
        
        combined_text = ". ".join(text_parts)
        
        return {
            "id": section_id,
            "section_num": section_num,
            "law_type": "BNS",
            "section_name": section_name,
            "description": description,
            "chapter": chapter,
            "chapter_name": chapter_name,
            "text": combined_text,
            "char_count": len(combined_text),
            "processed_at": datetime.now().isoformat()
        }
    
    def clean_bsa_dataset(self, filename: str = "bsa_sections.csv") -> str:
        """
        Clean the BSA (Bharatiya Sakshya Adhiniyam) dataset.
        Similar structure to BNS.
        """
        logger.info("Starting BSA dataset cleaning...")
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            logger.warning(f"BSA file not found: {filepath}")
            return ""
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Loaded {len(df)} BSA records")
        except Exception as e:
            logger.error(f"Error reading BSA: {e}")
            return ""
        
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        cleaned_records = []
        for idx, row in df.iterrows():
            try:
                # Reuse BNS cleaning logic with different law_type
                record = self._clean_bns_record(row, idx)
                if record:
                    record["id"] = record["id"].replace("BNS_", "BSA_")
                    record["law_type"] = "BSA"
                    record["text"] = record["text"].replace("BNS Section", "BSA Section")
                    cleaned_records.append(record)
                    self.stats["cleaned_records"] += 1
            except Exception as e:
                logger.warning(f"Error cleaning BSA row {idx}: {e}")
        
        output_path = self.processed_dir / "bsa_clean.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.success(f"Cleaned {len(cleaned_records)} BSA sections → {output_path}")
        return str(output_path)
    
    def print_stats(self):
        """Print cleaning statistics"""
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        print(f"Total records processed: {self.stats['total_records']}")
        print(f"Successfully cleaned:    {self.stats['cleaned_records']}")
        print(f"Skipped/Invalid:         {self.stats['skipped_records']}")
        if self.stats['errors']:
            print(f"Errors: {len(self.stats['errors'])}")
            for err in self.stats['errors']:
                print(f"  - {err}")
        print("="*50 + "\n")


def main():
    """Main entry point for data cleaning pipeline"""
    print("\n" + "="*60)
    print("NEETHI APP - DATA CLEANING PIPELINE (Step 1)")
    print("="*60 + "\n")
    
    # Initialize cleaner
    cleaner = LegalDataCleaner()
    
    # Clean available datasets
    ipc_output = cleaner.clean_ipc_dataset()
    bns_output = cleaner.clean_bns_dataset()
    bsa_output = cleaner.clean_bsa_dataset()
    
    # Print summary
    cleaner.print_stats()
    
    if ipc_output:
        print(f"✅ IPC data ready: {ipc_output}")
    if bns_output:
        print(f"✅ BNS data ready: {bns_output}")
    if bsa_output:
        print(f"✅ BSA data ready: {bsa_output}")
    
    if not any([ipc_output, bns_output, bsa_output]):
        print("\n⚠️  No datasets were processed!")
        print("Please download datasets and place in data/raw/:")
        print("  - ipc_sections.csv from Kaggle")
        print("  - bns_sections.csv from Kaggle")
        print("  - bsa_sections.csv from Kaggle")
    
    print("\nNext step: Run 02_chunking.py")


if __name__ == "__main__":
    main()
