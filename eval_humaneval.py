import argparse
import json
import jsonlines
import re
from vllm import LLM, SamplingParams
from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.evaluate import evaluate
import os
import sys
from datetime import datetime
import csv

MAX_INT = sys.maxsize

def extract_code(completion: str) -> str:
    """
    Extracts Python code from model completion containing markdown code blocks.

    Args:
        completion (str): Full model output text

    Returns:
        str: Extracted Python code without markdown formatting
    """
    # Match both ```python and ``` blocks
    code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', completion, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()  # Return last code block
    return completion  # Fallback to raw completion

def batch_data(data_list, batch_size=1):
    """Batch data into chunks of specified size (reused from GSM8K)"""
    # ... existing implementation from eval_gsm8k.py ...

def humaneval_test(
    model: str,
    data_path: str,
    n_samples: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    start: int = 0,
    end: int = MAX_INT,
    batch_size: int = 1,
    tensor_parallel_size: int = 1
) -> dict:
    """
    Evaluate model on HumanEval code generation task.

    Args:
        model: Path to model weights
        data_path: Path to HumanEval dataset
        n_samples: Number of code samples to generate per problem
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling
        start: Start index of dataset
        end: End index of dataset
        batch_size: Batch size for generation
        tensor_parallel_size: GPU parallelism

    Returns:
        dict: Evaluation results with metrics and samples
    """
    # Load HumanEval+ dataset
    dataset = get_human_eval_plus(data_path)
    problems = list(dataset.values())[start:end]

    # Create code generation prompts
    prompt_template = (
        "Below is an instruction that describes a programming task. "
        "Write a Python function that completes the task.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response: Here's the Python implementation:\n"
    )
    prompts = [prompt_template.format(instruction=prob["prompt"]) for prob in problems]

    # Duplicate prompts for multiple samples
    full_prompts = [p for p in prompts for _ in range(n_samples)]

    # Configure generation parameters
    stop_tokens = ["\n###", "\nclass", "\nif", "\ndef", "\nprint"]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=512,
        stop=stop_tokens
    )

    # Initialize LLM and generate code
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
    outputs = llm.generate(full_prompts, sampling_params)

    # Process and extract code
    samples = []
    for idx, output in enumerate(outputs):
        problem_id = idx // n_samples
        completion = output.outputs[0].text
        code = extract_code(completion)
        samples.append({
            "task_id": problems[problem_id]["task_id"],
            "solution": code,
            "completion": completion
        })

    # Evaluate with EvalPlus
    eval_results = evaluate(
        dataset=dataset,
        samples=samples,
        k=[1, n_samples]  # Calculate pass@1 and pass@k
    )

    # Format results
    return {
        "pass@1": eval_results["pass@1"],
        f"pass@{n_samples}": eval_results[f"pass@{n_samples}"],
        "total_problems": len(problems),
        "samples": samples,
        "eval_details": eval_results["details"]
    }

def save_results(results, output_dir):
    """
    Save HumanEval evaluation results with code-specific metrics.

    Args:
        results (dict): Contains pass rates, samples, and eval details
        output_dir (str): Directory to save results

    Creates:
        - output_dir/
            - samples.jsonl: All generated code samples
            - eval_details.json: Detailed test case results
            - summary.csv: Pass rates and metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save generated samples
    samples_path = os.path.join(output_dir, "samples.jsonl")
    with jsonlines.open(samples_path, "w") as f:
        for sample in results["samples"]:
            f.write(sample)

    # Save evaluation details
    details_path = os.path.join(output_dir, "eval_details.json")
    with open(details_path, "w") as f:
        json.dump(results["eval_details"], f, indent=2)

    # Save summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pass@1": results["pass@1"],
        f"pass@{results['n_samples']}": results[f"pass@{results['n_samples']}"],
        "total_problems": results["total_problems"],
        "samples_per_problem": results["n_samples"]
    }

    # Write/append to CSV
    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a" if file_exists else "w") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_data)

    print(f"Results saved to {output_dir}")
    print(f"- Code samples: {samples_path}")
    print(f"- Evaluation details: {details_path}")
    print(f"- Summary metrics: {summary_path}")

def parse_args():
    """Argument parser with code-specific parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int, default=MAX_INT)
    parser.add_argument("-b", "--batch_size", type=int, default=20)
    parser.add_argument("-t", "--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("-n", "--n_samples", type=int, default=20,
                       help="Number of code samples per problem")
    parser.add_argument("-o", "--output_dir", type=str,
                       help="Output directory for results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = humaneval_test(
        model=args.model,
        data_path=args.data_file,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Add n_samples to results for saving
    results["n_samples"] = args.n_samples

    # Handle output directory
    if not args.output_dir:
        if args.model.startswith('meta-llama/'):
            model_name = 'code/' + args.model.split('/')[-1]  # Use full repo name as path
        else:
            model_name = os.path.relpath(args.model, 'checkpoints/')
        args.output_dir = f'outputs/{model_name}'

    save_results(results, args.output_dir)