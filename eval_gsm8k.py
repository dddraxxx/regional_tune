import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import os
import csv
from datetime import datetime
MAX_INT = sys.maxsize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.lower().split('the answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

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


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "End your response with 'The answer is: [your answer]'.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step. "
    )
    print('promt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
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
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        y_pred = extract_answer_number(completion)
        is_correct = False
        if y_pred is not None:
            is_correct = float(y_pred) == float(prompt_answer)
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
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)

    # Return results for optional file output
    return {
        'accuracy': acc,
        'total_samples': len(result),
        'invalid_count': len(invalid_outputs),
        'detailed_results': detailed_results
    }

def save_results(results, output_dir, test_name="gsm8k"):
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
            model_rel_path = 'gsm8k/' + args.model.split('/')[-1]  # Use full repo name as path
        else:
            model_rel_path = os.path.relpath(args.model, 'checkpoints/')
        args.output_dir = f'outputs/{model_rel_path}'
    results = gsm8k_test(
        model=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Save results to directory if specified
    if args.output_dir:
        save_results(results, args.output_dir, test_name="gsm8k")
