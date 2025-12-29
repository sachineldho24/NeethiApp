"""
Dataset Downloader for Training Pipeline

Downloads legal datasets from Hugging Face for fine-tuning InLegalBERT:
- Aalap Instruction Dataset (21K examples)
- Indian Legal Texts Fine-tuning dataset
- Custom Q&A pairs from IndicLegalQA

Usage:
    python pipelines/offline/06_download_datasets.py [--test]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

# Configure logging
logger.add(
    "logs/dataset_download_{time}.log",
    rotation="10 MB",
    level="INFO"
)


@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    source: str
    subset: Optional[str] = None
    split: str = "train"
    output_file: str = ""
    
    def __post_init__(self):
        if not self.output_file:
            self.output_file = f"{self.name}.jsonl"


# Dataset configurations
DATASETS: List[DatasetConfig] = [
    DatasetConfig(
        name="aalap_instruction",
        source="opennyaiorg/aalap_instruction_dataset",
        split="train",
        output_file="aalap_train.jsonl"
    ),
    DatasetConfig(
        name="legal_texts_finetune",
        source="Techmaestro369/indian-legal-texts-finetuning",
        split="train",
        output_file="legal_texts.jsonl"
    ),
]


def check_dependencies():
    """Check if required packages are installed"""
    try:
        from datasets import load_dataset
        return True
    except ImportError:
        logger.error("datasets package not installed. Run: pip install datasets")
        return False


def download_dataset(config: DatasetConfig, output_dir: Path) -> int:
    """
    Download a single dataset from Hugging Face.
    
    Args:
        config: Dataset configuration
        output_dir: Directory to save output
        
    Returns:
        Number of examples downloaded
    """
    from datasets import load_dataset
    
    logger.info(f"Downloading {config.name} from {config.source}...")
    
    try:
        # Load dataset
        if config.subset:
            ds = load_dataset(config.source, config.subset, split=config.split)
        else:
            ds = load_dataset(config.source, split=config.split)
        
        # Save to JSONL
        output_path = output_dir / config.output_file
        count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in ds:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"✅ Saved {count} examples to {output_path}")
        return count
        
    except Exception as e:
        logger.error(f"❌ Failed to download {config.name}: {e}")
        return 0


def download_aalap(output_dir: Path, max_samples: Optional[int] = None) -> Dict[str, int]:
    """
    Download Aalap instruction dataset with task-specific processing.
    
    The Aalap dataset contains multiple legal tasks:
    - argument_generation: Generate arguments from case facts
    - issue_generation: Identify legal issues
    - summary: Summarize judgments
    - timeline: Create case timeline
    
    Args:
        output_dir: Directory to save output
        max_samples: Maximum samples per task (for testing)
        
    Returns:
        Dict with task names and counts
    """
    from datasets import load_dataset
    
    logger.info("Downloading Aalap instruction dataset...")
    
    try:
        ds = load_dataset("opennyaiorg/aalap_instruction_dataset", split="train")
        
        # Log the actual column names from the dataset
        logger.info(f"Aalap columns: {ds.column_names}")
        
        # Process and save
        task_counts = {}
        all_examples = []
        
        for i, example in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            
            # Save all original fields from the dataset
            normalized = dict(example)  # Copy all fields
            normalized["id"] = f"aalap_{i}"
            normalized["source"] = "aalap"
            
            # Try to identify task type from various possible fields
            task = (
                example.get("task_type") or 
                example.get("task") or 
                example.get("type") or
                "instruction"
            )
            normalized["task"] = task
            task_counts[task] = task_counts.get(task, 0) + 1
            
            all_examples.append(normalized)
        
        # Save all examples
        output_path = output_dir / "aalap_train.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Aalap dataset saved: {len(all_examples)} examples")
        logger.info(f"   Task breakdown: {task_counts}")
        
        # Show sample record
        if all_examples:
            sample_keys = list(all_examples[0].keys())
            logger.info(f"   Fields: {sample_keys}")
        
        return task_counts
        
    except Exception as e:
        logger.error(f"❌ Failed to download Aalap: {e}")
        return {}


def download_legal_texts(output_dir: Path, max_samples: Optional[int] = None) -> int:
    """
    Download Indian legal texts fine-tuning dataset.
    
    Contains Q&A pairs based on IPC, CrPC, and Constitution.
    Original source: Kaggle (Akshat Gupta)
    
    Note: The HuggingFace dataset has inconsistent schemas across JSON files,
    so we download each file separately and normalize.
    
    Args:
        output_dir: Directory to save output
        max_samples: Maximum samples (for testing)
        
    Returns:
        Number of examples downloaded
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return 0
    
    logger.info("Downloading Indian legal texts dataset (individual files)...")
    
    repo_id = "Techmaestro369/indian-legal-texts-finetuning"
    files = ["ipc_qa.json", "crpc_qa.json", "constitution_qa.json"]
    
    all_examples = []
    
    for filename in files:
        try:
            # Download individual JSON file
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine source from filename
            source = filename.replace("_qa.json", "").upper()
            
            # Normalize each example
            for i, item in enumerate(data):
                if max_samples and len(all_examples) >= max_samples:
                    break
                
                normalized = {
                    "id": f"legal_texts_{source}_{i}",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "source": source,
                    "section": item.get("section", "")
                }
                all_examples.append(normalized)
            
            logger.info(f"  ✅ {filename}: {len(data)} examples")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to download {filename}: {e}")
    
    if not all_examples:
        logger.error("❌ No legal texts downloaded")
        return 0
    
    # Save all examples
    output_path = output_dir / "legal_texts.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Legal texts dataset saved: {len(all_examples)} examples")
    return len(all_examples)


def create_sample_data(output_dir: Path) -> int:
    """
    Create sample training data for testing when HuggingFace is unavailable.
    
    Returns:
        Number of examples created
    """
    logger.info("Creating sample training data...")
    
    samples = [
        {
            "id": "sample_1",
            "question": "What is the punishment for theft under IPC?",
            "answer": "Under Section 379 of IPC (now Section 303 of BNS), theft is punishable with imprisonment up to 3 years, or with fine, or with both.",
            "context": "Section 379 IPC - Punishment for theft.",
            "source": "sample"
        },
        {
            "id": "sample_2",
            "question": "Can police arrest without a warrant?",
            "answer": "Police can arrest without warrant only for cognizable offenses as per Section 41 CrPC. For non-cognizable offenses, a warrant from Magistrate is required.",
            "context": "Section 41 CrPC - When police may arrest without warrant.",
            "source": "sample"
        },
        {
            "id": "sample_3",
            "question": "What is anticipatory bail?",
            "answer": "Anticipatory bail is a direction to release a person on bail in case of arrest for a non-bailable offense. It can be granted under Section 438 CrPC (now Section 482 BNSS).",
            "context": "Section 438 CrPC - Direction for grant of bail to person apprehending arrest.",
            "source": "sample"
        },
        {
            "id": "sample_4",
            "question": "What is FIR and who can file it?",
            "answer": "FIR (First Information Report) is the first document filed when police receive information about a cognizable offense. Any person with knowledge of the offense can file it under Section 154 CrPC.",
            "context": "Section 154 CrPC - Information in cognizable cases.",
            "source": "sample"
        },
        {
            "id": "sample_5",
            "question": "What is the difference between IPC and BNS?",
            "answer": "BNS (Bharatiya Nyaya Sanhita) is the new criminal code that replaced the IPC (Indian Penal Code) from July 2024. It reorganizes offenses, updates definitions, and includes new offenses like organized crime and terrorism.",
            "context": "BNS replaced IPC as the primary criminal law of India.",
            "source": "sample"
        }
    ]
    
    output_path = output_dir / "sample_qa.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Sample data created: {len(samples)} examples")
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Download sample data only (for testing)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to download per dataset"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    if args.test:
        # Just create sample data
        count = create_sample_data(output_dir)
        logger.info(f"Test mode: Created {count} sample examples")
        return
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Creating sample data instead.")
        create_sample_data(output_dir)
        return
    
    # Download datasets
    total = 0
    
    # Aalap
    task_counts = download_aalap(output_dir, args.max_samples)
    total += sum(task_counts.values())
    
    # Legal texts
    count = download_legal_texts(output_dir, args.max_samples)
    total += count
    
    # Create sample data as fallback
    create_sample_data(output_dir)
    
    logger.info(f"\n📊 Download Summary:")
    logger.info(f"   Total examples: {total}")
    logger.info(f"   Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
