"""
Training Data Preparation for InLegalBERT Fine-tuning

Generates training triplets for contrastive learning:
- (query, positive_chunk, negative_chunk)

The model learns:
- Positive: "This answer IS relevant to this question"
- Negative: "This other answer is NOT relevant"

Usage:
    python pipelines/offline/07_prepare_training.py [--sample] [--input-dir data/training]
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from collections import defaultdict

# Configure logging
logger.add(
    "logs/training_prep_{time}.log",
    rotation="10 MB",
    level="INFO"
)


@dataclass
class Triplet:
    """Training triplet for contrastive learning"""
    query: str
    positive: str
    negative: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "positive": self.positive,
            "negative": self.negative,
            "metadata": self.metadata or {}
        }


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_triplets(triplets: List[Triplet], output_path: Path):
    """Save triplets to JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet.to_dict(), ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(triplets)} triplets to {output_path}")


def generate_triplets_from_qa(
    qa_pairs: List[Dict],
    negatives_per_positive: int = 3,
    min_text_length: int = 20
) -> List[Triplet]:
    """
    Generate triplets from Q&A pairs.
    
    Strategy:
    - Query: The question
    - Positive: The correct answer
    - Negative: Random answer from different question
    """
    triplets = []
    
    # Filter valid pairs
    valid_pairs = [
        p for p in qa_pairs 
        if len(p.get("answer", p.get("output", ""))) >= min_text_length
        and len(p.get("question", p.get("instruction", ""))) >= min_text_length
    ]
    
    if len(valid_pairs) < 2:
        logger.warning("Not enough valid Q&A pairs for triplet generation")
        return []
    
    # Create answer pool for negative sampling
    answer_pool = [
        p.get("answer", p.get("output", ""))
        for p in valid_pairs
    ]
    
    for i, pair in enumerate(valid_pairs):
        query = pair.get("question", pair.get("instruction", ""))
        positive = pair.get("answer", pair.get("output", ""))
        
        # Sample random negatives (excluding current answer)
        other_answers = [a for j, a in enumerate(answer_pool) if j != i]
        
        if len(other_answers) >= negatives_per_positive:
            negatives = random.sample(other_answers, negatives_per_positive)
        else:
            negatives = other_answers
        
        # Create triplets
        for neg in negatives:
            triplet = Triplet(
                query=query,
                positive=positive,
                negative=neg,
                metadata={
                    "source": pair.get("source", "unknown"),
                    "id": pair.get("id", f"pair_{i}")
                }
            )
            triplets.append(triplet)
    
    return triplets


def generate_triplets_from_aalap(
    aalap_data: List[Dict],
    negatives_per_positive: int = 2
) -> List[Triplet]:
    """
    Generate triplets from Aalap instruction data.
    
    Aalap fields:
    - input_text: Case facts or context
    - user_prompt: The instruction/question
    - output_text: The expected response
    - combined_input_prompt: Full prompt with system + user
    - task: Task type (e.g., argument_generation, issue_generation)
    """
    triplets = []
    
    # Group by task type
    by_task = defaultdict(list)
    for item in aalap_data:
        task = item.get("task", "unknown")
        by_task[task].append(item)
    
    logger.info(f"Aalap tasks: {dict((k, len(v)) for k, v in by_task.items())}")
    
    for task, items in by_task.items():
        if len(items) < 2:
            continue
        
        # Get output pool for this task (for hard negatives - same task, wrong answer)
        # Aalap uses 'output_text' not 'output' - handle None values
        output_pool = [
            item.get("output_text") or ""
            for item in items
            if len(item.get("output_text") or "") >= 20
        ]
        
        for i, item in enumerate(items):
            # Query = user_prompt or combined_input_prompt
            query = item.get("user_prompt") or item.get("combined_input_prompt") or ""
            
            # Positive = output_text (the correct response)
            positive = item.get("output_text") or ""
            
            if len(positive) < 20 or len(query) < 20:
                continue
            
            # Sample hard negatives (same task type, different output)
            other_outputs = [o for j, o in enumerate(output_pool) if j != i]
            
            if len(other_outputs) >= negatives_per_positive:
                negatives = random.sample(other_outputs, negatives_per_positive)
            else:
                negatives = other_outputs[:negatives_per_positive]
            
            for neg in negatives:
                triplet = Triplet(
                    query=query[:1000],  # Limit query length
                    positive=positive[:1000],
                    negative=neg[:1000],
                    metadata={
                        "source": "aalap",
                        "task": task,
                        "id": item.get("id", f"aalap_{i}")
                    }
                )
                triplets.append(triplet)
    
    return triplets


def generate_combined_triplets(
    input_dir: Path,
    negatives_per_positive: int = 3
) -> List[Triplet]:
    """
    Generate triplets from all available datasets.
    """
    all_triplets = []
    
    # Process sample Q&A
    sample_qa_path = input_dir / "sample_qa.jsonl"
    if sample_qa_path.exists():
        logger.info("Processing sample Q&A data...")
        qa_data = load_jsonl(sample_qa_path)
        triplets = generate_triplets_from_qa(qa_data, negatives_per_positive)
        all_triplets.extend(triplets)
        logger.info(f"Generated {len(triplets)} triplets from sample Q&A")
    
    # Process legal texts
    legal_texts_path = input_dir / "legal_texts.jsonl"
    if legal_texts_path.exists():
        logger.info("Processing legal texts data...")
        legal_data = load_jsonl(legal_texts_path)
        triplets = generate_triplets_from_qa(legal_data, negatives_per_positive)
        all_triplets.extend(triplets)
        logger.info(f"Generated {len(triplets)} triplets from legal texts")
    
    # Process Aalap
    aalap_path = input_dir / "aalap_train.jsonl"
    if aalap_path.exists():
        logger.info("Processing Aalap instruction data...")
        aalap_data = load_jsonl(aalap_path)
        triplets = generate_triplets_from_aalap(aalap_data, negatives_per_positive)
        all_triplets.extend(triplets)
        logger.info(f"Generated {len(triplets)} triplets from Aalap")
    
    return all_triplets


def create_sample_triplets(output_path: Path) -> int:
    """
    Create sample triplets for testing.
    """
    sample_triplets = [
        Triplet(
            query="What is the punishment for theft under IPC?",
            positive="Under Section 379 of IPC, theft is punishable with imprisonment up to 3 years, or with fine, or with both.",
            negative="Murder under Section 302 IPC is punishable with death or life imprisonment.",
            metadata={"source": "sample", "type": "theft_vs_murder"}
        ),
        Triplet(
            query="What is the punishment for theft under IPC?",
            positive="Under Section 379 of IPC, theft is punishable with imprisonment up to 3 years, or with fine, or with both.",
            negative="Defamation under Section 499 IPC can lead to imprisonment up to 2 years.",
            metadata={"source": "sample", "type": "theft_vs_defamation"}
        ),
        Triplet(
            query="Can police arrest without a warrant?",
            positive="Police can arrest without warrant only for cognizable offenses as per Section 41 CrPC.",
            negative="Theft is punishable with imprisonment up to 3 years or fine.",
            metadata={"source": "sample", "type": "arrest_vs_theft"}
        ),
        Triplet(
            query="What is anticipatory bail?",
            positive="Anticipatory bail is a direction to release on bail in case of arrest for a non-bailable offense under Section 438 CrPC.",
            negative="FIR is the first document filed when police receive information about a cognizable offense.",
            metadata={"source": "sample", "type": "bail_vs_fir"}
        ),
    ]
    
    save_triplets(sample_triplets, output_path)
    return len(sample_triplets)


def main():
    parser = argparse.ArgumentParser(description="Prepare training triplets")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/training",
        help="Input directory with datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for triplets"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create sample triplets only (for testing)"
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=3,
        help="Number of negatives per positive example"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "training_triplets.jsonl"
    
    if args.sample:
        # Just create sample triplets
        count = create_sample_triplets(output_path)
        logger.info(f"Sample mode: Created {count} sample triplets")
        return
    
    # Check if input directory exists
    if not input_dir.exists():
        logger.warning(f"Input directory {input_dir} not found. Creating sample triplets.")
        create_sample_triplets(output_path)
        return
    
    # Generate triplets from available data
    triplets = generate_combined_triplets(input_dir, args.negatives)
    
    if not triplets:
        logger.warning("No triplets generated. Creating sample triplets.")
        create_sample_triplets(output_path)
        return
    
    # Shuffle triplets
    random.shuffle(triplets)
    
    # Save triplets
    save_triplets(triplets, output_path)
    
    # Print summary
    logger.info(f"\n📊 Training Data Summary:")
    logger.info(f"   Total triplets: {len(triplets)}")
    logger.info(f"   Output file: {output_path}")
    
    # Source breakdown
    sources = defaultdict(int)
    for t in triplets:
        source = t.metadata.get("source", "unknown") if t.metadata else "unknown"
        sources[source] += 1
    logger.info(f"   By source: {dict(sources)}")


if __name__ == "__main__":
    main()
