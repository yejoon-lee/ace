#!/usr/bin/env python3
import os
import re
import json
import openai
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

def initialize_clients(
    api_provider: str = "openai",
    generator_base_url: str = None,
    generator_api_key: str = None,
    reflector_base_url: str = None,
    reflector_api_key: str = None,
    curator_base_url: str = None,
    curator_api_key: str = None,
):
    """Initialize separate clients for generator, reflector, and curator.

    If per-actor base_url/api_key are provided, they override the api_provider defaults.
    """
    provider_urls = {
        "sambanova": "https://api.sambanova.ai/v1",
        "together": "https://api.together.xyz/v1",
        "openai": "https://api.openai.com/v1",
    }
    provider_keys = {
        "sambanova": os.getenv('SAMBANOVA_API_KEY', ''),
        "together": os.getenv('TOGETHER_API_KEY', ''),
        "openai": os.getenv('OPENAI_API_KEY', ''),
    }
    default_url = provider_urls[api_provider]
    default_key = provider_keys[api_provider]

    gen_url = generator_base_url or default_url
    gen_key = generator_api_key or default_key
    ref_url = reflector_base_url or default_url
    ref_key = reflector_api_key or default_key
    cur_url = curator_base_url or default_url
    cur_key = curator_api_key or default_key

    generator_client = openai.OpenAI(api_key=gen_key, base_url=gen_url)
    reflector_client = openai.OpenAI(api_key=ref_key, base_url=ref_url)
    curator_client = openai.OpenAI(api_key=cur_key, base_url=cur_url)

    print(f"Generator client: {gen_url}")
    print(f"Reflector client: {ref_url}")
    print(f"Curator client: {cur_url}")
    return generator_client, reflector_client, curator_client

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None


def _normalize_extracted_answer(answer: Any) -> str:
    """Normalize extracted answer text for stable downstream scoring."""
    text = str(answer).strip()
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
    ):
        text = text[1:-1].strip()
    return text


def extract_answer(response):
    """Extract final_answer from model response with strict-first JSON parsing."""
    if response is None:
        return "No final answer found"
    if not isinstance(response, str):
        response = str(response)

    try:
        # 1) Strict JSON parse of the full response.
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "final_answer" in parsed:
            answer = _normalize_extracted_answer(parsed.get("final_answer", ""))
            if answer:
                return answer

    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    # 2) Try JSON recovery from fenced or braced snippets.
    json_candidates: list[str] = []
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, flags=re.DOTALL)
    if fenced_match:
        json_candidates.append(fenced_match.group(1))
    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        json_candidates.append(response[first_brace:last_brace + 1])
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "final_answer" in parsed:
                answer = _normalize_extracted_answer(parsed.get("final_answer", ""))
                if answer:
                    return answer
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
            continue

    # 3) Regex fallbacks.
    matches = re.findall(r"Finish\[(.*?)\]", response)
    if matches:
        answer = _normalize_extracted_answer(matches[-1])
        if answer:
            return answer

    matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
    if matches:
        answer = _normalize_extracted_answer(matches[-1])
        if answer:
            return answer

    matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
    if matches:
        answer = _normalize_extracted_answer(matches[-1])
        if answer:
            return answer

    # Handle simple unquoted values: "final_answer": some_value
    matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}\n]+)', response)
    if matches:
        answer = _normalize_extracted_answer(matches[-1])
        answer = re.sub(r'[,}]*$', '', answer).strip()
        if answer:
            return answer

    final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
    match = re.search(final_answer_pattern, response)
    if match:
        remaining_text = response[match.start():]
        boxed_content = extract_boxed_content(remaining_text)
        if boxed_content:
            answer = _normalize_extracted_answer(boxed_content)
            if answer:
                return answer

    matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
    if matches:
        answer = _normalize_extracted_answer(matches[-1])
        answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
        answer = answer.replace('$', '').strip()
        if answer:
            return answer

    return "No final answer found"
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


def evaluate_single_test_sample(args_tuple, data_processor) -> Tuple[Dict, str]:
    """
    Evaluate a single test sample - task-agnostic implementation.
    
    Args:
        args_tuple: Tuple of (index, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode)
        data_processor: DataProcessor instance with answer_is_correct method
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        gen_response, bullet_ids, call_info = generator.generate(  # BOOKMARK: test in ace eventually leads to here
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"test_eval_{i}",
            log_dir=log_dir
        )

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return {
            "index": i,
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "success": True
        }, None

    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20, 
                      use_json_mode=False) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of test set - task-agnostic implementation.
    
    Args:
        data_processor: DataProcessor instance with answer_is_correct and evaluate_accuracy methods
        generator: Generator instance
        playbook: Current playbook string
        test_samples: List of test samples
        max_tokens: Max tokens for generation
        log_dir: Directory for logs
        max_workers: Number of parallel workers
        use_json_mode: Whether to use JSON mode
        
    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]

    results = {
        "correct": 0, "total": 0, "no_answer": 0,
        "answers": [], "targets": [], "errors": []
    }

    # Use a wrapper to pass data_processor to the evaluation function
    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(eval_wrapper, args): args 
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            result, error = future.result()
            
            if error:
                print(error)
                continue

            if result and result["success"]:
                results["correct"] += (1 if result["is_correct"] else 0)
                results["total"] += 1
                results["answers"].append(result["final_answer"])
                results["targets"].append(result["target"])
                
                if not result["is_correct"]:
                    results["errors"].append({
                        "index": result["index"],
                        "prediction": result["final_answer"],
                        "ground_truth": result["target"]
                    })
                
                if result["final_answer"] == "No final answer found":
                    results["no_answer"] += 1

            if i % 50 == 0:
                curr_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {curr_acc:.3f}")
    
    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(results["answers"], results["targets"])
        
        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"]
        }
        
        error_logs = {
            "accuracy": accuracy,
            "errors": results["errors"]
        }
        
        print(f"\n📊 Final Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")
    else:
        results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print(f"\n📊 No valid results!")
        
    return final_results, error_logs