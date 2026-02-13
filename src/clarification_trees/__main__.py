import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from clarification_trees.dialog_tree import DialogTree, NodeType
from clarification_trees.dataset import ClearVQADataset
from clarification_trees.utils import add_cq_messages, add_answer_messages, add_inference_messages
from src.clarification_trees.models.vllm.remote_vllm_model import RemoteVLLMModel

app = typer.Typer()

async def _start_clarification_server(clarification_model_cfg: DictConfig, lora_checkpoint_path: Path, clarification_model_gpus: list[int], clarification_model_port: int, clarification_model_log_file: Path, environment_path: Path):
    model = RemoteVLLMModel(
        clarification_model_cfg,
        lora_checkpoint_path,
        gpus=clarification_model_gpus,
        port=clarification_model_port,
        log_file=clarification_model_log_file,
        environment_path=environment_path
    )
    print(f"Starting clarification server on port {clarification_model_port}")
    await model.initialize_server()
    print(f"Clarification server started on port {clarification_model_port}")
    return model

async def _start_answer_server(answer_model_cfg: DictConfig, lora_checkpoint_path: Path, answer_model_gpus: list[int], answer_model_port: int, answer_model_log_file: Path, environment_path: Path):
    model = RemoteVLLMModel(
        answer_model_cfg,
        lora_checkpoint_path,
        gpus=answer_model_gpus,
        port=answer_model_port,
        log_file=answer_model_log_file,
        environment_path=environment_path
    )
    print(f"Starting answer server on port {answer_model_port}")
    await model.initialize_server()
    print(f"Answer server started on port {answer_model_port}")
    return model

@asynccontextmanager
async def _start_servers(cfg: DictConfig, environment_path: Path):
    lora_checkpoint_path = Path(cfg.paths.checkpoints.loras)
    clarification_model_cfg = cfg.clarification_model
    answer_model_cfg = cfg.answer_model
    clarification_model_gpus = cfg.devices.clarification
    answer_model_gpus = cfg.devices.answer
    clarification_model_port = cfg.remote_vllm.clarification.port
    answer_model_port = cfg.remote_vllm.answer.port
    clarification_model_log_file = Path(cfg.remote_vllm.clarification.log_file)
    answer_model_log_file = Path(cfg.remote_vllm.answer.log_file)
    
    print(typer.style("Starting clarification model with config:", fg=typer.colors.GREEN, bold=True))
    print(OmegaConf.to_yaml(clarification_model_cfg))
    print(typer.style("Starting answer model with config:", fg=typer.colors.GREEN, bold=True))
    print(OmegaConf.to_yaml(answer_model_cfg))

    print(typer.style("Generic Configuration:", fg=typer.colors.GREEN, bold=True))
    print(f"  LoRA Checkpoint Path: {lora_checkpoint_path}")
    print(f"  Clarification Model GPUs: {clarification_model_gpus}")
    print(f"  Answer Model GPUs: {answer_model_gpus}")
    print(f"  Clarification Model Port: {clarification_model_port}")
    print(f"  Answer Model Port: {answer_model_port}")
    print(f"  Clarification Model Log File: {clarification_model_log_file}")
    print(f"  Answer Model Log File: {answer_model_log_file}")

    clarification_model, answer_model = await asyncio.gather(
        _start_clarification_server(
            clarification_model_cfg,
            lora_checkpoint_path,
            clarification_model_gpus,
            clarification_model_port,
            clarification_model_log_file,
            environment_path
        ),
        _start_answer_server(
            answer_model_cfg,
            lora_checkpoint_path,
            answer_model_gpus,
            answer_model_port,
            answer_model_log_file,
            environment_path
        )
    )

    # Spin until interrupted and then kill the servers
    try:
        yield clarification_model, answer_model
    finally:
        print("Shutting down servers...")
        clarification_model.stop_server()
        answer_model.stop_server()

@app.command()
def vllm_serve(
    ctx: typer.Context, 
    config_name: str = typer.Option("config", help="Name of the config file to use"),
    environment_path: Path = typer.Option(None, help="Path to the environment to use")
):
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name, overrides=ctx.args)

    async def serve_and_spin():
        async with _start_servers(cfg, environment_path):
            while True:
                await asyncio.sleep(1)
    
    asyncio.run(serve_and_spin())


@app.command()
def test_vllm_server(
    ctx: typer.Context,
    config_name: str = typer.Option("config", help="Name of the config file to use"),
    environment_path: Path = typer.Option(None, help="Path to the environment to use")
):
    """
    Starts a loop that allows the user to test the vLLM servers.
    """
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name, overrides=ctx.args)

    async def test_servers():
        ds = ClearVQADataset(load_images=False)
        async with _start_servers(cfg, environment_path) as (clarification_model, answer_model):
            while True:
                print(typer.style("\n\n--------------------", fg=typer.colors.GREEN, bold=True))
                print(typer.style("Enter a sample index for testing", fg=typer.colors.GREEN, bold=True))
                sample_index = await asyncio.to_thread(input, "Sample Index: ")
                sample_index = int(sample_index)
                
                sample = ds[sample_index]
                tree = DialogTree(
                    sample.blurred_question,
                    None,
                    sample.image_path,
                    sample.caption,
                    sample.question,
                    sample.gold_answer,
                    sample.answers
                )

                print(typer.style("\n******* Sample *******", fg=typer.colors.GREEN, bold=True))
                print(f"Unambiguous Question: {sample.question}")
                print(f"Ambiguous Question: {sample.blurred_question}")
                print(f"Gold Answer: {sample.gold_answer}")
                print(f"Answers: {sample.answers}")
                print(f"Image Path: {sample.image_path}")
                print(f"Caption: {sample.caption}")
                print(typer.style("**********************\n", fg=typer.colors.GREEN, bold=True))

                clarification_input_node = DialogTree.ROOT

                for _ in range(5):
                    # Get a clarifying question from the clarification model
                    dialog_traj = tree.get_trajectory(clarification_input_node)
                    messages = dialog_traj.to_messages("qwen-3-vl", use_img_path=True)
                    add_cq_messages(messages, cfg=cfg)
                    # print(f"Testing clarification model with messages:\n{messages}")

                    clarification_response_obj = await clarification_model.generate(messages, use_lora=True)
                    clarification_response = clarification_response_obj.choices[0].message.content
                    assert clarification_response is not None
                    print(typer.style("\nClarification Response:", fg=typer.colors.RED, bold=True))
                    print(clarification_response)
                    answer_input_node = tree.add_node(clarification_input_node, NodeType.CLARIFICATION_QUESTION, clarification_response)

                    # Get an answer from the answer model
                    dialog_traj = tree.get_trajectory(answer_input_node)
                    messages = dialog_traj.to_messages("qwen-3-vl", use_img_path=True)
                    add_answer_messages(messages, sample.question, sample.answers, cfg=cfg)
                    # print(f"Testing answer model with messages:\n{messages}")

                    answer_response_obj = await answer_model.generate(messages, use_lora=False)
                    answer_response = answer_response_obj.choices[0].message.content
                    assert answer_response is not None
                    print(typer.style("\nAnswer Response:", fg=typer.colors.BRIGHT_BLUE, bold=True))
                    print(answer_response)
                    clarification_input_node = tree.add_node(answer_input_node, NodeType.CLARIFYING_ANSWER, answer_response)

                    # Get an inference from the answer model
                    dialog_traj = tree.get_trajectory(clarification_input_node)
                    messages = dialog_traj.to_messages("qwen-3-vl", use_img_path=True)
                    add_inference_messages(messages, cfg=cfg)
                    # print(f"Testing answer model with messages:\n{messages}")

                    inference_response_obj = await answer_model.generate(messages, use_lora=False)
                    inference_response = inference_response_obj.choices[0].message.content
                    assert inference_response is not None
                    print(typer.style("\n>> Inference Response:", fg=typer.colors.BRIGHT_GREEN, bold=True))
                    print(">> " + inference_response)
                    tree.add_node(clarification_input_node, NodeType.INFERENCE, inference_response)

                


    asyncio.run(test_servers())

    


if __name__ == "__main__":
    # ignore_unknown_options=True allows us to pass Hydra overrides (like db.host=...)
    # without Typer throwing an "Unknown argument" error.
    app()