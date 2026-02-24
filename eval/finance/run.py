#!/usr/bin/env python3
"""
Example usage script for the ACE system.

Modified for DiaKV
Examples:
export PYTHONPATH=/home/yejoon/kv/ace:$PYTHONPATH
python -m eval.finance.run \
    --task_name finer \
    --mode offline \
    --save_path results \
    --api_provider openai \
    --generator_model Qwen/Qwen3-4B-Instruct-2507 \
    --generator_base_url http://localhost:8000/v1 \
    --generator_api_key EMPTY \
    --generator_api_provider vllm \
    --generator_max_tokens 4096 \
    --reflector_model gpt-5.2-2025-12-11 \
    --reflector_max_tokens 16384 \
    --curator_model gpt-5.2-2025-12-11 \
    --curator_max_tokens 16384 \
    --max_num_rounds 2 \
    --save_steps N \
    --eval_steps N

python -m eval.finance.run \
    --task_name finer \
    --mode offline \
    --save_path results \
    --api_provider openai \
    --generator_model gpt-5-mini \
    --reflector_model gpt-5.2 \
    --curator_model gpt-5.2 \
    --generator_max_tokens 16384 \
    --reflector_max_tokens 16384 \
    --curator_max_tokens 16384

Without --initial_playbook_path, --use_bulletpoint_analyzer, and --bulletpoint_analyzer_threshold, configs to ACE is idential to baseline/ace/run.py
"""
import os
import json
import openai
import argparse
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple
from .data_processor import DataProcessor

from ace import ACE
from llm import timed_llm_call
from utils import initialize_clients


SIMPLE_EVAL_SYSTEM_PROMPT: str = "Answer the question with a concise and accurate answer. Only return the answer, no other text."

SIMPLE_EVAL_PROMPT: str = (
    "{context}"
    " Answer the question with a concise and accurate answer. Only return the answer, no other text."
    "\n\nPlaybook:\n{playbook}"
    "\n\n- Read the playbook carefully and apply relevant strategies, formulas, and insights"
    "\n- Pay attention to common mistakes listed in the playbook and avoid them"
    "\n\nQuestion: {question}"
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ACE System - Refactored')
    
    # Task configuration
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task (e.g., 'finer', 'formula')")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Path to initial playbook (optional)")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode: 'offline' for offline training with validation, "
                             "'online' for online training and testing on test split, "
                             "'eval_only' for testing only with provided playbook")
    
    # Model configuration
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai"], help="API provider")
    parser.add_argument("--generator_model", type=str, 
                        default="DeepSeek-V3.1",
                        help="Model for generator")
    parser.add_argument("--reflector_model", type=str,
                        default="DeepSeek-V3.1",
                        help="Model for reflector")
    parser.add_argument("--curator_model", type=str,
                        default="DeepSeek-V3.1",
                        help="Model for curator")
    
    # Per-role overrides (base_url, api_key, api_provider, max_tokens)
    for role in ("generator", "reflector", "curator"):
        parser.add_argument(f"--{role}_base_url", type=str, default=None,
                            help=f"Override base URL for {role} client")
        parser.add_argument(f"--{role}_api_key", type=str, default=None,
                            help=f"Override API key for {role} client")
        parser.add_argument(f"--{role}_api_provider", type=str, default=None,
                            help=f"Override api_provider for {role}")
        parser.add_argument(f"--{role}_max_tokens", type=int, default=None,
                            help=f"Override max_tokens for {role}")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_num_rounds", type=int, default=2,
                        help="Max reflection rounds for incorrect answers")
    parser.add_argument("--curator_frequency", type=int, default=1,
                        help="Run curator every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--online_eval_frequency", type=int, default=15,
                        help="Update playbook every N samples for evaluation in online mode")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save intermediate playbooks every N steps")
    
    # System configuration
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens for LLM responses")
    parser.add_argument("--playbook_token_budget", type=int, default=80000,
                        help="Total token budget for playbook")
    parser.add_argument("--test_workers", type=int, default=50,
                        help="Number of parallel workers for testing")
    
    # Prompt configuration
    parser.add_argument("--json_mode", action="store_true",
                        help="Enable JSON mode for LLM calls")
    parser.add_argument("--no_ground_truth", action="store_true",
                        help="Don't use ground truth in reflection")
    
    # Bulletpoint analyzer configuration
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true",
                        help="Enable bulletpoint analyzer for deduplication and merging")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90,
                        help="Similarity threshold for bulletpoint analyzer (0-1, default: 0.90)")
    
    # Output configuration
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")
    
    return parser.parse_args()

def load_data(data_path: str):
    """
    Load and process data from a JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data

def preprocess_data(task_name, config, mode):
    """
    Load training and test data for the specified task.
    
    Args:
        task_name: Name of the task
        config: Configuration dictionary with data paths
        mode: Run mode ('offline', 'online', or 'eval_only')
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples, data_processor)
        - For offline mode: all three are loaded
        - For online mode: only test_samples
        - For eval_only mode: only test_samples
    """
    processor = DataProcessor(task_name=task_name)
    
    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")
        
        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")
    
    # For offline mode, load train, val, and optionally test data
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []
        
        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")
    
    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def simple_eval_playbook(
    ace_system: ACE,
    playbook: str,
    test_samples: List[Dict[str, Any]],
    data_processor: "DataProcessor",
    save_dir: str,
    prefix: str = "test",
) -> Dict[str, Any]:
    """Evaluate playbook on test samples using a simplified prompt.

    Uses a stripped-down prompt instead of ACE's internal eval, which
    reuses the complex reflection/curation prompts that burden weaker LMs.

    Args:
        ace_system: ACE instance (used for its generator client).
        playbook: Playbook text to evaluate.
        test_samples: List of dicts with 'context', 'question', 'target'.
        data_processor: DataProcessor with evaluate_accuracy().
        save_dir: Directory to save results JSON.
        prefix: Filename prefix (e.g. 'initial', 'final').

    Returns:
        Results dict with 'accuracy', 'correct', 'total'.
    """
    gen = ace_system.generator
    predictions: List[str] = []
    targets: List[str] = []

    for idx, sample in enumerate(test_samples):
        prompt = SIMPLE_EVAL_PROMPT.format(
            context=sample["context"],
            playbook=playbook,
            question=sample["question"],
        )
        response, _ = timed_llm_call(
            gen.api_client,
            gen.api_provider,
            gen.model,
            prompt,
            role="simple_eval",
            call_id=f"{prefix}_{idx}",
            max_tokens=gen.max_tokens,
            system_prompt=SIMPLE_EVAL_SYSTEM_PROMPT,
        )
        predictions.append(response.strip())
        targets.append(sample["target"])

    accuracy = data_processor.evaluate_accuracy(predictions, targets)
    correct = int(round(accuracy * len(test_samples)))
    results: Dict[str, Any] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_samples),
        "no_answer": 0,
    }

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"{prefix}_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  {prefix.upper()} Test Accuracy: {accuracy:.3f}")
    return results


def simple_eval_validation_fn(
    data_processor: "DataProcessor",
) -> Callable[..., Tuple[Dict, Dict]]:
    """Return a validation callback that uses the simplified eval prompt.

    Callback signature matches ACE's validation_eval_fn:
        (ace_system, playbook, val_samples, save_path, log_dir, epoch, step) -> (val_results, error_log)
    """
    def validation_callback(ace_system, playbook, val_samples, save_path, log_dir, epoch, step):
        gen = ace_system.generator
        predictions: List[str] = []
        targets: List[str] = []

        for idx, sample in enumerate(val_samples):
            prompt = SIMPLE_EVAL_PROMPT.format(
                context=sample["context"],
                playbook=playbook,
                question=sample["question"],
            )
            response, _ = timed_llm_call(
                gen.api_client,
                gen.api_provider,
                gen.model,
                prompt,
                role="simple_eval",
                call_id=f"val_e{epoch}_s{step}_{idx}",
                max_tokens=gen.max_tokens,
                system_prompt=SIMPLE_EVAL_SYSTEM_PROMPT,
            )
            predictions.append(response.strip())
            targets.append(sample["target"])

        accuracy = data_processor.evaluate_accuracy(predictions, targets)
        correct = int(round(accuracy * len(val_samples)))
        val_results: Dict[str, Any] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(val_samples),
            "no_answer": 0,
        }
        error_log: Dict[str, Any] = {"accuracy": accuracy, "errors": []}
        print(f"  Val (simple_eval): acc={accuracy:.3f}")
        return val_results, error_log

    return validation_callback


def main():
    """Main execution function."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'='*60}\n")
    
    # Load data
    with open("./eval/finance/data/sample_config.json", 'r') as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, 
        task_config[args.task_name],
        args.mode
    )
        
    # Load initial playbook (or use empty if None provided)
    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")
    
    # Create ACE system
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        generator_max_tokens=args.generator_max_tokens,
        reflector_max_tokens=args.reflector_max_tokens,
        curator_max_tokens=args.curator_max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
        generator_base_url=args.generator_base_url,
        generator_api_key=args.generator_api_key,
        generator_api_provider=args.generator_api_provider,
        reflector_base_url=args.reflector_base_url,
        reflector_api_key=args.reflector_api_key,
        reflector_api_provider=args.reflector_api_provider,
        curator_base_url=args.curator_base_url,
        curator_api_key=args.curator_api_key,
        curator_api_provider=args.curator_api_provider,
    )
    
    # Prepare configuration
    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'eval_steps': args.eval_steps,
        'online_eval_frequency': args.online_eval_frequency,
        'save_steps': args.save_steps,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
        'api_provider': args.api_provider
    }
    
    # --- Simple eval for initial / final test, bypassing ACE's internal eval ---
    os.makedirs(args.save_path, exist_ok=True)

    if args.mode == "eval_only":
        # No training; just evaluate with the loaded playbook.
        playbook = initial_playbook or ""
        print(f"\n{'='*60}")
        print(f"EVAL ONLY (simple_eval)")
        print(f"{'='*60}\n")
        simple_eval_playbook(
            ace_system, playbook, test_samples, data_processor,
            save_dir=args.save_path, prefix="eval_only",
        )
    else:
        # Initial eval (before training) with empty / initial playbook
        if test_samples:
            print(f"\n{'='*60}")
            print(f"INITIAL TEST (simple_eval, before training)")
            print(f"{'='*60}\n")
            simple_eval_playbook(
                ace_system, ace_system.playbook, test_samples, data_processor,
                save_dir=args.save_path, prefix="initial",
            )

        # Run ACE training.
        # test_samples=None so ACE skips its internal initial/final eval;
        # validation_eval_fn overrides the default ACE validation prompt.
        ace_system.run(
            mode=args.mode,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=None,
            data_processor=data_processor,
            config=config,
            validation_eval_fn=simple_eval_validation_fn(data_processor),
        )

        # Final eval (after training) with best playbook
        if test_samples:
            print(f"\n{'='*60}")
            print(f"FINAL TEST (simple_eval, with best playbook)")
            print(f"{'='*60}\n")
            simple_eval_playbook(
                ace_system, ace_system.best_playbook, test_samples, data_processor,
                save_dir=args.save_path, prefix="final",
            )


if __name__ == "__main__":
    main()