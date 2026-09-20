import logging
import pandas as pd
logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)

import click
from pathlib import Path
from config import Config, DataPath
from types import SimpleNamespace
from typing import Optional
import fetch, build, evaluation, baselines, pipeline

# Dynamically resolve the default path: src -> project_root -> config -> cfg.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "cfg.yaml"

@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    help="Path to the main YAML configuration file."
)
@click.pass_context
def cli(ctx, config: Path):
    """MTG Benchmark Data and Experiment CLI."""
    if ctx.obj is None:
        ctx.obj = SimpleNamespace()
    
    try:
        # Now you can assign directly using dot notation
        ctx.obj.cfg = Config.load(config)    
    except FileNotFoundError:
        click.secho(f"Error: Configuration file not found at {config}", fg="red")
        ctx.exit(1)
    except Exception as e:
        click.secho(f"Error loading configuration: {e}", fg="red")
        ctx.exit(1)


@cli.command()
@click.option("--set-code", type=str, default=None, help="Specific set to download. If omitted, downloads all.")
@click.pass_context
def download(ctx, set_code: Optional[str]=None):
    """Download MTG draft dataset for a specific set code (e.g., otj, mkm)."""
    cfg = ctx.obj.cfg  # Access the fully loaded Config object
    datapath = DataPath(cfg)
    if set_code:
        fetch.download_set_code(datapath, set_code)    
    else:
        fetch.download_all_set_codes(datapath)
    

@cli.command()
@click.argument("set_code", type=str)
@click.option("--split", type=int, default=None)
@click.pass_context
def build_reference_effects(ctx, set_code, split=None):
    cfg = ctx.obj.cfg
    build.build_reference_effects(cfg, set_code, split=split)
    

@cli.command()
@click.argument("set_code", type=str)
@click.argument("methods_path", type=str)
@click.option("--idx", type=int, default=None)
@click.option("--split", type=int, default=None)
@click.option("--filter", type=str, default=None)
@click.pass_context
def build_baseline_estimates(ctx, set_code, methods_path, idx=None, split=None, filter=None):
    method_names = read_names_from_file(methods_path)
    pipeline.build_baseline_estimates(ctx.obj.cfg, set_code, method_names, idx=idx, split=split, filter=filter)

@cli.command()
@click.argument("set_code", type=str)
@click.pass_context
def independence_diagnostic(ctx, set_code):
    pipeline.independence_diagnostic(ctx.obj.cfg, set_code)

@cli.command()
@click.argument("set_code", type=str)
@click.pass_context
def instrument_diagnostic(ctx, set_code):
    pipeline.instrument_diagnostic(ctx.obj.cfg, set_code)


@cli.command()
@click.argument("set_code", type=str)
@click.option("--split", type=int, default=None)
@click.pass_context
def extract_baseline_features(ctx, set_code, split=None):
    pipeline.build_extractor(ctx.obj.cfg, set_code, split=split)

@cli.command()
@click.argument("set_code", type=str)
@click.argument("profile", type=str)
@click.argument("space_path", type=str, default=None)
@click.pass_context
def hp_tuning_sample(ctx, set_code, profile, space_path=None):
    pipeline.hp_tuning_sample(ctx.obj.cfg, set_code, profile)


@cli.command()
@click.argument("topic", type=str)
@click.argument("set_codes", type=str)
@click.argument("methods_path", type=str)
@click.option("--split", type=int, default=None)
@click.pass_context
def create_summary(ctx, topic, set_codes, methods_path, split=None):
    set_codes = set_codes.split(',')
    method_names = read_names_from_file(methods_path)
    pipeline.summarize_results(ctx.obj.cfg, topic, set_codes, 
                               method_names=method_names, split=split)


@cli.command()
def list_available_estimators():
    import dragonnet
    estimator_names = [x.name for x in baselines.default_estimators()]
    estimator_names.append(dragonnet.CATE_NAME)
    print("Available estimators:")
    for name in estimator_names:
        print(name)

def read_names_from_file(file_path):
    names = []  
    # Open the file safely using 'with', ensuring it closes automatically
    with open(file_path, 'rt') as file:
        for line in file:
            name = line.strip()  # Removes spaces and newlines from both ends
            if name and not name.startswith('#'):             # Evaluates to False if the string is empty
                names.append(name)                
    return names

#     
if __name__ == "__main__":
    cli()