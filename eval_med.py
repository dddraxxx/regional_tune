import argparse
import json
import re
import jsonlines
from vllm import LLM, SamplingParams
import sys
import os
import csv
from datetime import datetime
MAX_INT = sys.maxsize

def extract_answer_option(completion):
    """Extract the answer option (A/B/C/D) from model completion.

    Args:
        completion (str): The model's output text

    Returns:
        str or None: The extracted answer option (A/B/C/D) or None if no valid answer found

    Example:
        >>> extract_answer_option("After analyzing the case... The answer is: D")
        'D'
        >>> extract_answer_option("The best option would be... The answer is: B.")
        'B'
        >>> extract_answer_option("The answer is: D. Known liver neoplasm")
        'D'
        >>> extract_answer_option("Let me explain... The answer is: E")
        None
    """
    text = completion.lower().split('the answer is: ')
    if len(text) > 1:
        # Get first character after "the answer is:"
        answer = text[-1].strip()[0]
        # Check if it's a valid option A-D
        if answer.lower() in ['a', 'b', 'c', 'd']:
            return answer.upper()
    return None

def batch_data(data_list, batch_size=1):
    """Batch data into chunks of specified size."""
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def med_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    """Evaluate model on medical multiple choice questions.

    Args:
        model (str): Path to the model or model identifier
        data_path (str): Path to the test data file
        start (int, optional): Starting index for evaluation. Defaults to 0.
        end (int, optional): Ending index for evaluation. Defaults to MAX_INT.
        batch_size (int, optional): Batch size for evaluation. Defaults to 1.
        tensor_parallel_size (int, optional): Tensor parallel size. Defaults to 1.

    Returns:
        dict: Evaluation results containing accuracy and detailed outputs

    Example input data format:
        {
            "Question": "A junior orthopaedic surgery resident...",
            "Options": {
                "A": "Disclose the error...",
                "B": "Refuse to dictate...",
                "C": "Report the physician...",
                "D": "Tell the attending..."
            },
            "Correct Answer": "Tell the attending...",
            "Correct Option": "D"
        }
    """
    med_ins = []
    med_answers = []
    problem_prompt = (
        "Below is a question that describes a task. "
        "Write a response that appropriately completes the request."
        "End your response with 'The answer is: [your answer, a single letter]'.\n\n"
        "### Question:\n{instruction}\n\n### Response: "
    )
    print('prompt =====', problem_prompt)

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # Format question with options
            instruction = f"{item['Question']}\n\nChoices:\nA. {item['Options']['A']}\nB. {item['Options']['B']}\nC. {item['Options']['C']}\nD. {item['Options']['D']}"
            temp_instr = problem_prompt.format(instruction=instruction)
            med_ins.append(temp_instr)
            med_answers.append(item['Correct Option'])

    med_ins = med_ins[start:end]
    med_answers = med_answers[start:end]
    print('length ====', len(med_ins))
    batch_med_ins = batch_data(med_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=128, stop=stop_tokens)
    print('sampling =====', sampling_params)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
    result = []
    res_completions = []

    for idx, (prompt, prompt_answer) in enumerate(zip(batch_med_ins, med_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    detailed_results = []  # Store detailed results for each question
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(med_ins, res_completions, med_answers)):
        y_pred = extract_answer_option(completion)
        is_correct = False
        if y_pred is not None:
            is_correct = y_pred == prompt_answer
            result.append(is_correct)
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

        # Store detailed result for each question
        detailed_results.append({
            'question': prompt,
            'predicted': y_pred,
            'true_answer': prompt_answer,
            'model_output': completion,
            'is_correct': is_correct,
        })

    acc = sum(result) / len(result)
    print(f'Invalid outputs: {len(invalid_outputs)} out of {len(result)} total samples')
    print('start===', start, ', end====', end)
    print('med length====', len(result), ', med acc====', acc)

    return {
        'accuracy': acc,
        'total_samples': len(result),
        'invalid_count': len(invalid_outputs),
        'detailed_results': detailed_results
    }

def save_results(results, output_dir, test_name="med"):
    """Save evaluation results to a directory structure.

    Args:
        results (dict): Dictionary containing evaluation results
        output_dir (str): Directory path to save results
        test_name (str): Name of the test (e.g., 'med') for file naming

    Creates:
        - output_dir/
            - detailed_results_{test_name}.json: Full evaluation details
            - summary.csv: Basic metrics in CSV format with test_name column
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results as JSON with test name
    detailed_path = os.path.join(output_dir, f'detailed_results_{test_name}.json')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary as CSV
    summary_path = os.path.join(output_dir, 'summary.csv')
    summary_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_name': test_name,
        'accuracy': results['accuracy'],
        'total_samples': results['total_samples'],
        'invalid_count': results['invalid_count'],
        'valid_count': results['total_samples'] - results['invalid_count']
    }

    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_data)

    print(f"\nResults saved to directory: {output_dir}")
    print(f"- Detailed results: {detailed_path}")
    print(f"- Summary CSV: {summary_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    parser.add_argument("-o", "--output_dir", type=str, help="Directory to save evaluation results (optional)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.output_dir:
        if args.model.startswith('meta-llama/'):
            model_rel_path = 'med/' + args.model.split('/')[-1]
        else:
            model_rel_path = os.path.relpath(args.model, 'checkpoints/')
        args.output_dir = f'outputs/{model_rel_path}'

    results = med_test(
        model=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Save results to directory if specified
    if args.output_dir:
        save_results(results, args.output_dir, test_name="med")