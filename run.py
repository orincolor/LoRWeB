# Copyright (C) 2026 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

import argparse
from jobs import ExtensionJob
from toolkit.config import get_config
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()

def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )

    # ADDITIONAL ARGUMENTS
    parser.add_argument('--name', type=str, default=None) #config.name, config.process.logging.run_name
    parser.add_argument('--linear', type=int, default=None) #config.process.network.linear
    parser.add_argument('--linear_alpha', type=int, default=None) #config.process.network.linear_alpha
    parser.add_argument('--loras_num', type=int, default=None) #config.process.network.network_kwargs.loras_num
    parser.add_argument('--lora_keys_dim', type=int, default=None) #config.process.network.network_kwargs.lora_keys_dim
    parser.add_argument('--lora_heads', type=int, default=None) #config.process.network.network_kwargs.lora_heads
    parser.add_argument('--lora_softmax', type=str, default=None) #config.process.network.network_kwargs.lora_softmax
    parser.add_argument('--mixing_coeffs_type', type=str, default=None) #config.process.network.network_kwargs.mixing_coeffs_type
    parser.add_argument('--pooling_type', type=str, default=None) #config.process.network.network_kwargs.pooling_type
    parser.add_argument('--query_projection_type', type=str, default=None) #config.process.network.network_kwargs.query_projection_type
    parser.add_argument('--external_query_model', type=str, default=None) #config.process.network.network_kwargs.external_query_model
    parser.add_argument('--query_mode', type=str, default=None) #config.process.network.network_kwargs.query_mode
    parser.add_argument('--ignore_if_contains', type=list, default=None) #config.process.network.network_kwargs.ignore_if_contains
    parser.add_argument('--folder_path', type=str, default=None) #config.process.datasets.folder_path
    parser.add_argument('--control_path', type=str, default=None) #config.process.datasets.control_path
    parser.add_argument('--batch_size', type=int, default=None) #config.process.train.batch_size
    parser.add_argument('--steps', type=int, default=None) #config.process.train.steps
    parser.add_argument('--lr', type=float, default=None) #config.process.train.lr
    parser.add_argument('--lr_scheduler', type=str, default=None) #config.process.train.lr_scheduler
    parser.add_argument('--disable_wandb', dest='use_wandb', action='store_false', default=True) #config.process.logging.use_wandb
    parser.add_argument('--resume_id', type=str, default=None) #config.process.logging.resume_id
    parser.add_argument('--cache_latents_to_disk', action='store_true', default=False) #config.process.datasets.cache_latents_to_disk

    parser.add_argument('--is_bidirectional_analogy', action='store_true', default=False) #config.process.datasets.is_bidirectional_analogy
    parser.add_argument('--save_every', type=int, default=None) #config.process.save.save_every

    parser.add_argument('--gradient_accumulation_steps', type=int, default=None) #config.process.train.gradient_accumulation_steps
    parser.add_argument('--gradient_accumulation', type=int, default=None) #config.process.train.gradient_accumulation
    parser.add_argument('--max_grad_norm', type=float, default=None) #config.process.train.max_grad_norm

    parser.add_argument('--debug', action='store_true', default=False)


    ARG_TO_CONFIG_PATH = {
        'name': 'config.name',
        'linear': 'config.process.network.linear',
        'linear_alpha': 'config.process.network.linear_alpha',
        'loras_num': 'config.process.network.network_kwargs.loras_num',
        'lora_keys_dim': 'config.process.network.network_kwargs.lora_keys_dim',
        'lora_heads': 'config.process.network.network_kwargs.lora_heads',
        'lora_softmax': 'config.process.network.network_kwargs.lora_softmax',
        'mixing_coeffs_type': 'config.process.network.network_kwargs.mixing_coeffs_type',
        'pooling_type': 'config.process.network.network_kwargs.pooling_type',
        'query_projection_type': 'config.process.network.network_kwargs.query_projection_type',
        'external_query_model': 'config.process.network.network_kwargs.external_query_model',
        'query_mode': 'config.process.network.network_kwargs.query_mode',
        'ignore_if_contains': 'config.process.network.network_kwargs.ignore_if_contains',
        'folder_path': 'config.process.datasets.folder_path',
        'control_path': 'config.process.datasets.control_path',
        'batch_size': 'config.process.train.batch_size',
        'steps': 'config.process.train.steps',
        'lr_scheduler': 'config.process.train.lr_scheduler',
        'lr': 'config.process.train.lr',
        'gradient_accumulation_steps': 'config.process.train.gradient_accumulation_steps',
        'gradient_accumulation': 'config.process.train.gradient_accumulation',
        'use_wandb': 'config.process.logging.use_wandb',
        'resume_id': 'config.process.logging.resume_id',
        'cache_latents_to_disk': 'config.process.datasets.cache_latents_to_disk',
        'max_grad_norm': 'config.process.train.max_grad_norm',
        'save_every': 'config.process.save.save_every',
        'is_bidirectional_analogy': 'config.process.datasets.is_bidirectional_analogy',
    }

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(("0.0.0.0", 5679))
        debugpy.wait_for_client()

    # convert lora_softmax to bool
    if args.lora_softmax is not None:
        args.lora_softmax = args.lora_softmax.lower() == 'true'

    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    # build from the args the structured kwargs dict
    kwargs = {}
    for key, value in vars(args).items():
        if value is not None and key in ARG_TO_CONFIG_PATH:
            # Build nested structure from the path
            path = ARG_TO_CONFIG_PATH[key]
            parts = path.split('.')
            current = kwargs
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
    if kwargs:
        kwargs['config']['process'] = [kwargs['config']['process']]
        if 'datasets' in kwargs['config']['process'][0]:
            kwargs['config']['process'][0]['datasets'] = [kwargs['config']['process'][0]['datasets']]

    if args.name is not None:
        if 'logging' not in kwargs['config']['process'][0]:
            kwargs['config']['process'][0]['logging'] = {}
        kwargs['config']['process'][0]['logging']['run_name'] = args.name

    print("Overriding config with:", kwargs)

    for config_file in config_file_list:
        try:
            config = get_config(config_file, None, kwargs)
            job = ExtensionJob(config)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e


if __name__ == '__main__':
    main()
