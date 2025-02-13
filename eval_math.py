import argparse
import json
import pdb
import jsonlines
import os
import csv
from datetime import datetime

import util
from vllm import LLM, SamplingParams
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        is_correct = util.is_equiv(extract_ans, answer)
        return {
            'extracted_answer': extract_ans,
            'is_correct': is_correct,
            'is_invalid': False
        }
    else:
        return {
            'extracted_answer': None,
            'is_correct': False,
            'is_invalid': True
        }

def batch_data(data_list, batch_size=1):
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

def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt_temp = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    detailed_results = []
    invalid_count = 0
    correct_count = 0

    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        result = process_results(prompt, completion, prompt_answer)
        if result['is_invalid']:
            invalid_count += 1
        if result['is_correct']:
            correct_count += 1

        detailed_results.append({
            'question': prompt,
            'model_output': completion,
            'true_answer': prompt_answer,
            'predicted': result['extracted_answer'],
            'is_correct': result['is_correct'],
            'is_invalid': result['is_invalid']
        })

    total_samples = len(detailed_results)
    accuracy = correct_count / total_samples

    print('Invalid outputs:', invalid_count)
    print('start===', start, ', end====', end)
    print('Total samples:', total_samples, ', Accuracy:', accuracy)

    return {
        'accuracy': accuracy,
        'total_samples': total_samples,
        'invalid_count': invalid_count,
        'correct_count': correct_count,
        'detailed_results': detailed_results
    }

def save_results(results, output_dir, test_name="math"):
    """
    Save evaluation results to a directory structure.

    Args:
        results (dict): Dictionary containing evaluation results
        output_dir (str): Directory path to save results
        test_name (str): Name of the test (e.g., 'math', 'gsm8k') for file naming

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
        'correct_count': results['correct_count'],
        'valid_count': results['total_samples'] - results['invalid_count']
    }

    # Check if file exists to determine if we need to write headers
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
    parser.add_argument("--model", type=str, default='')  # model path
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
            model_rel_path = 'math/' + args.model.split('/')[-1]  # Use full repo name as path
        else:
            model_rel_path = os.path.relpath(args.model, 'checkpoints/')
        args.output_dir = f'outputs/{model_rel_path}'
    results = test_hendrycks_math(
        model=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Save results to directory if specified
    if args.output_dir:
        save_results(results, args.output_dir, test_name="math")
