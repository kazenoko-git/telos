"""Small, standalone Gemma 4 E4B inference check for Kaggle's two T4s.

No Télos adapter, quantization, benchmark downloads, or generated-code execution.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--output', default='/kaggle/working/gemma4_e4b_smoke.json')
    args = parser.parse_args()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Select GPU T4 x2 in Kaggle before running this test.')
    if args.max_new_tokens < 1:
        raise ValueError('--max-new-tokens must be positive')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_id = 'google/gemma-4-E4B-it'
    revision = HfApi().model_info(model_id, revision=args.revision).sha
    report = {
        'kind': 'inference_smoke_test_not_benchmark',
        'model': model_id, 'revision': revision,
        'requested_dtype': args.dtype, 'quantization': None,
        'thinking': args.thinking, 'do_sample': False,
        'max_new_tokens': args.max_new_tokens,
        'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in
                     ['torch', 'transformers', 'accelerate', 'huggingface-hub']},
        'cuda_runtime': torch.version.cuda,
        'gpus': [{'name': torch.cuda.get_device_name(i),
                  'total_bytes': torch.cuda.get_device_properties(i).total_memory,
                  'capability': list(torch.cuda.get_device_capability(i))}
                 for i in range(2)],
        'results': [],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        temporary.replace(destination)

    def sync():
        for i in range(2):
            torch.cuda.synchronize(i)

    save()
    try:
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        start = time.perf_counter()
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, args.dtype),
            device_map='balanced', max_memory={0: '13GiB', 1: '13GiB', 'cpu': '8GiB'},
            offload_folder='/kaggle/working/gemma4_offload',
            attn_implementation='eager',
        ).eval()
        sync()
        report['load_seconds'] = time.perf_counter() - start
        placement = getattr(model, 'hf_device_map', {})
        report['device_map'] = {k: str(v) for k, v in placement.items()}
        report['cpu_or_disk_offload'] = any(str(v) in ('cpu', 'disk') for v in placement.values())
        report['parameter_elements_by_dtype'] = dict(Counter())
        for parameter in model.parameters():
            key = str(parameter.dtype)
            report['parameter_elements_by_dtype'][key] = (
                report['parameter_elements_by_dtype'].get(key, 0) + parameter.numel())
        print('Placement:', report['device_map'])
        print('Parameter dtypes:', report['parameter_elements_by_dtype'])
        if not all(any(str(v) in (str(i), f'cuda:{i}') for v in placement.values()) for i in range(2)):
            raise RuntimeError('Model placement did not use both GPUs; inspect the saved device map.')
        save()

        prompts = [
            ('arithmetic', 'What is 17 * 19? Reply with only the integer.'),
            ('json', 'Return only valid JSON with two keys: planet set to Earth and moons set to 1.'),
            ('code', 'Write a Python function add(a, b) that returns their sum. Return only the function.'),
        ]
        input_device = model.get_input_embeddings().weight.device
        if input_device.type == 'meta':
            input_device = torch.device('cuda:0')
        for name, prompt in prompts:
            for i in range(2):
                torch.cuda.reset_peak_memory_stats(i)
            inputs = processor.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=True, return_dict=True,
                return_tensors='pt', add_generation_prompt=True,
                enable_thinking=args.thinking,
            ).to(input_device)
            sync()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    return_dict_in_generate=True, output_scores=True,
                )
            sync()
            seconds = time.perf_counter() - start
            new_ids = output.sequences[0, inputs['input_ids'].shape[-1]:]
            raw = processor.decode(new_ids, skip_special_tokens=False)
            parsed = processor.parse_response(raw, prefix=inputs['input_ids'])
            # Negative infinity can be an intentional generation mask.
            valid_scores = all(
                not torch.isnan(s).any().item() and not torch.isposinf(s).any().item()
                and torch.isfinite(s).any(dim=-1).all().item() for s in output.scores
            )
            eos = model.generation_config.eos_token_id
            eos_ids = [eos] if isinstance(eos, int) else (eos or [])
            result = {
                'name': name, 'prompt': prompt, 'raw_output': raw, 'parsed_output': parsed,
                'input_tokens': inputs['input_ids'].shape[-1],
                'generated_tokens': len(new_ids), 'seconds': seconds,
                'tokens_per_second_including_prefill': len(new_ids) / seconds,
                'hit_token_limit': len(new_ids) >= args.max_new_tokens and int(new_ids[-1]) not in eos_ids,
                'scores_valid': valid_scores,
                'gpu_peak_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)],
                'gpu_peak_reserved_bytes': [torch.cuda.max_memory_reserved(i) for i in range(2)],
            }
            report['results'].append(result)
            save()
            print(json.dumps(result, indent=2, default=str))
            if not valid_scores:
                raise RuntimeError(f'Invalid generation scores for {name}; report saved.')
            del output
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        # Avoid persisting exception strings that could include authentication details.
        report['error_type'] = type(exc).__name__
        raise
    finally:
        save()
        print(f'Report: {destination}')


if __name__ == '__main__':
    main()
