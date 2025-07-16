# SPDX-License-Identifier: Apache-2.0
"""Benchmark offline inference throughput."""
import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any, Optional, Union

import torch
import uvloop
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators
from torch.cuda import cudart
from pyinstrument import Profiler
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.cuda.nvtx as nvtx


def run_vllm(
    requests: list[list[int]],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
    output_len: Optional[int] = None,
) -> tuple[float, Optional[list[RequestOutput]]]:
    from vllm import LLM, SamplingParams
    llm = LLM(**dataclasses.asdict(engine_args))
    # assert all(
    #     llm.llm_engine.model_config.max_model_len >= (
    #         request.prompt_len + request.expected_output_len)
    #     for request in requests), (
    #         "Please ensure that max_model_len is greater than the sum of"
    #         " prompt_len and expected_output_len for all requests.")
    # Add the requests to the engine.
    sampling_params: list[SamplingParams] = []
    for request in requests:
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
                detokenize=not disable_detokenize,
            ))
    lora_requests: Optional[list[LoRARequest]] = None
    
    use_beam_search = False

    outputs = None
    if not use_beam_search:
        start = time.perf_counter()
        nvtx.range_push("target_domain")
        outputs = llm.generate(prompt_token_ids=requests,
                               sampling_params=sampling_params,
                               lora_request=lora_requests,
                               use_tqdm=True)
        nvtx.range_pop()
        end = time.perf_counter()

    return end - start, outputs



async def run_vllm_async(
    requests: list[list[int]],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
) -> float:
    from vllm import SamplingParams

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        assert all(
            llm.model_config.max_model_len >= (request.prompt_len +
                                               request.expected_output_len)
            for request in requests), (
                "Please ensure that max_model_len is greater than the sum of"
                " prompt_len and expected_output_len for all requests.")

        # Add the requests to the engine.
        prompts: list[Union[TextPrompt, TokensPrompt]] = []
        sampling_params: list[SamplingParams] = []
        lora_requests: list[Optional[LoRARequest]] = []
        for request in requests:
            prompts.append(
                TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"],
                        multi_modal_data=request.multi_modal_data)
                if "prompt_token_ids" in request.prompt else \
                TextPrompt(prompt=request.prompt,
                           multi_modal_data=request.multi_modal_data))
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                    detokenize=not disable_detokenize,
                ))
            lora_requests.append(request.lora_request)

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp,
                lr) in enumerate(zip(prompts, sampling_params, lora_requests)):
            generator = llm.generate(prompt,
                                     sp,
                                     lora_request=lr,
                                     request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()
        return end - start



def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k]
            for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    sample_kwargs = {
        "tokenizer": tokenizer,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    # sample_kwargs["range_ratio"] = args.random_range_ratio

    # vocab = tokenizer.get_vocab()
    # special_tokens = tokenizer.special_tokens_map
    # requests = []
    # while len(requests) < args.num_prompts:
    #     request = []
    #     while len(request) + 2 < args.input_len:
    #         token = random.choice(list(vocab.keys()))
    #         if token not in special_tokens.values():
    #             token_id = tokenizer.convert_tokens_to_ids(token)
    #             request.append(token_id)
    #     request.insert(0, tokenizer.bos_token_id)
    #     request.append(tokenizer.eos_token_id)
    #     requests.append(request)
    # return requests
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.vocab_size  # 获取实际词汇表大小
    special_tokens = tokenizer.special_tokens_map

    # 验证特殊 token 的 ID 有效性
    def validate_token_id(token_name):
        token_id = getattr(tokenizer, f"{token_name}_token_id", None)
        if token_id is None or token_id >= vocab_size or token_id < 0:
            raise ValueError(
                f"Invalid {token_name}_token_id: {token_id}. "
                f"Must be in range [0, {vocab_size-1}]"
            )
        return token_id

    bos_token_id = validate_token_id("bos")
    eos_token_id = validate_token_id("eos")

    requests = []
    valid_tokens = [
        token for token in vocab.keys() 
        if token not in special_tokens.values()
    ]

    while len(requests) < args.num_prompts:
        request = []
        while len(request) + 2 < args.input_len:
            token = random.choice(valid_tokens)
            token_id = tokenizer.convert_tokens_to_ids(token)
            
            # 确保 ID 在有效范围内
            if 0 <= token_id < vocab_size:
                request.append(token_id)
            else:
                # 跳过无效 token
                continue
        
        # 插入已验证的特殊 token
        request.insert(0, bos_token_id)
        request.append(eos_token_id)
        requests.append(request)
    return requests
    


def main(args: argparse.Namespace):
    if args.seed is None:
        args.seed = 0
    print(args)
    random.seed(args.seed)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = get_requests(args, tokenizer)
    request_outputs: Optional[list[RequestOutput]] = None
    if args.async_engine:
        elapsed_time = uvloop.run(
            run_vllm_async(
                requests,
                args.n,
                AsyncEngineArgs.from_cli_args(args),
                args.disable_frontend_multiprocessing,
                args.disable_detokenize,
            ))
    else:
        elapsed_time, request_outputs = run_vllm(
            requests, args.n, EngineArgs.from_cli_args(args),
            args.disable_detokenize, args.output_len)


    if request_outputs:
        # Note: with the vllm and vllm-chat backends,
        # we have request_outputs, which we use to count tokens.
        total_prompt_tokens = 0
        total_output_tokens = 0
        for ro in request_outputs:
            if not isinstance(ro, RequestOutput):
                continue
            total_prompt_tokens += len(
                ro.prompt_token_ids) if ro.prompt_token_ids else 0
            total_output_tokens += sum(
                len(o.token_ids) for o in ro.outputs if o)
        total_num_tokens = total_prompt_tokens + total_output_tokens
    else:
        total_num_tokens = sum(r.prompt_len + r.expected_output_len
                               for r in requests)
        total_output_tokens = sum(r.expected_output_len for r in requests)
        total_prompt_tokens = total_num_tokens - total_output_tokens

    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
          f"{total_output_tokens / elapsed_time:.2f} output tokens/s")
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")
    print(f"Total infer-time: {elapsed_time}s")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize the response (i.e. do not include "
              "detokenization time in the measurement)"))
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=None,
        help="Range of sampled ratio of input/output length, "
        "used only for RandomDataSet.",
    )

    
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    # validate_args(args)
    
    # profiler = Profiler()
    # profiler.start()


    torch.cuda.profiler.start()
    main(args)
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()  # 确保同步
    # profiler.stop()
    # with open("profile.html", "w") as f:
    #     f.write(profiler.output_html())
