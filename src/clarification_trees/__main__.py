import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from clarification_trees.dialog_tree import DialogTree, NodeType, TreeSidecar
from clarification_trees.dataset import ClearVQADataset
from clarification_trees.utils import add_cq_messages, add_answer_messages, add_inference_messages, get_judge_messages, processes_judge_response
from clarification_trees import utils
from clarification_trees.models.vllm.remote_vllm_model import RemoteVLLMModel

app = typer.Typer()

async def _start_clarification_server(
    clarification_model_cfg: DictConfig, 
    lora_checkpoint_path: Path, 
    clarification_model_gpus: list[int], 
    clarification_model_gpu_memory_utilization: float,
    clarification_model_max_lora_rank: int,
    clarification_model_max_model_len: int,
    clarification_model_port: int, 
    clarification_model_log_file: Path, 
    environment_path: Path
):
    model = RemoteVLLMModel(
        clarification_model_cfg,
        lora_checkpoint_path,
        gpu_memory_utilization=clarification_model_gpu_memory_utilization,
        max_lora_rank=clarification_model_max_lora_rank,
        max_model_len=clarification_model_max_model_len,
        gpus=clarification_model_gpus,
        port=clarification_model_port,
        log_file=clarification_model_log_file,
        environment_path=environment_path
    )
    print(f"Starting clarification server on port {clarification_model_port}")
    await model.initialize_server()
    print(f"Clarification server started on port {clarification_model_port}")
    return model

async def _start_answer_server(
    answer_model_cfg: DictConfig, 
    lora_checkpoint_path: Path, 
    answer_model_gpus: list[int], 
    answer_model_gpu_memory_utilization: float,
    answer_model_max_lora_rank: int,
    answer_model_max_model_len: int,
    answer_model_port: int, 
    answer_model_log_file: Path, 
    environment_path: Path
):
    model = RemoteVLLMModel(
        answer_model_cfg,
        lora_checkpoint_path,
        gpu_memory_utilization=answer_model_gpu_memory_utilization,
        max_lora_rank=answer_model_max_lora_rank,
        max_model_len=answer_model_max_model_len,
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
    clarification_model_gpu_memory_utilization = cfg.remote_vllm.clarification.gpu_memory_utilization
    answer_model_gpu_memory_utilization = cfg.remote_vllm.answer.gpu_memory_utilization
    clarification_model_max_lora_rank = cfg.remote_vllm.clarification.max_lora_rank
    answer_model_max_lora_rank = cfg.remote_vllm.answer.max_lora_rank
    clarification_model_max_model_len = cfg.remote_vllm.clarification.max_model_len
    answer_model_max_model_len = cfg.remote_vllm.answer.max_model_len
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
            clarification_model_gpu_memory_utilization,
            clarification_model_max_lora_rank,
            clarification_model_max_model_len,
            clarification_model_port,
            clarification_model_log_file,
            environment_path
        ),
        _start_answer_server(
            answer_model_cfg,
            lora_checkpoint_path,
            answer_model_gpus,
            answer_model_gpu_memory_utilization,
            answer_model_max_lora_rank,
            answer_model_max_model_len,
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

                async def _generate_inference_and_scores(tree: DialogTree, clarification_input_node: int):
                    dialog_traj = tree.get_trajectory(clarification_input_node)
                    messages = dialog_traj.to_messages("qwen-3-vl", use_img_path=True)
                    add_inference_messages(messages, cfg=cfg)
                    inference_response_obj = await answer_model.generate(messages, use_lora=False)
                    inference_response = inference_response_obj.choices[0].message.content
                    assert inference_response is not None
                    print(typer.style("\n>> Inference Response:", fg=typer.colors.BRIGHT_GREEN, bold=True))
                    print(">> " + inference_response)
                    tree.add_node(clarification_input_node, NodeType.INFERENCE, inference_response)

                    # Get scores
                    messages = get_judge_messages(
                        unambiguous_question=sample.question,
                        gold_answer=sample.gold_answer,
                        answers=sample.answers,
                        caption=sample.caption,
                        inference_response=inference_response,
                        cfg=cfg
                    )
                    scores_response_obj = await answer_model.generate(messages, use_lora=False, n_outputs=10)
                    all_scores = []
                    for scores_response_obj in scores_response_obj.choices:
                        scores_response = scores_response_obj.message.content
                        assert scores_response is not None
                        reasoning, score = processes_judge_response(scores_response)
                        all_scores.append(score)
                    print(typer.style("\n>> Scores:", fg=typer.colors.BRIGHT_GREEN, bold=True))
                    print(f">> Reasoning: {reasoning}")
                    print(f">> Score: {all_scores} = {sum(all_scores)/len(all_scores)}")

                # Immediately try to get an inference
                await _generate_inference_and_scores(tree, clarification_input_node)

                ['vllm', 'serve', 'Qwen/Qwen3-VL-32B-Instruct', '--host', '0.0.0.0', '--port', '29003', '--trust-remote-code', '--tensor-parallel-size', '2', '--max-model-len', '4096', '--gpu-memory-utilization', '0.95', '--allowed-local-media-path', '/', '--enable-lora', '--max-lora-rank', '16']

                for _ in range(5):
                    # Get a clarifying question from the clarification model
                    dialog_traj = tree.get_trajectory(clarification_input_node)
                    messages = dialog_traj.to_messages("qwen-3-vl", use_img_path=True)
                    add_cq_messages(messages, cfg=cfg)
                    # print(f"Testing clarification model with messages:\n{messages}")

                    clarification_response_obj = await clarification_model.generate(messages, use_lora=True, logprobs=True, use_tokens_as_ids=True)
                    """
                    clarification_response_obj.choices[0].logprobs.content
                    [ChatCompletionTokenLogprob(token='Are', bytes=[65, 114, 101], logprob=-0.007280366960912943, top_logprobs=[]), ChatCompletionTokenLogprob(token=' you', bytes=[32, 121, 111, 117], logprob=-6.425174069590867e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token=' asking', bytes=[32, 97, 115, 107, 105, 110, 103], ...ob=-0.019521024078130722, top_logprobs=[]), ChatCompletionTokenLogprob(token=' about', bytes=[32, 97, 98, 111, 117, 116], logprob=-0.0012260308722034097, top_logprobs=[]), ChatCompletionTokenLogprob(token=' the', bytes=[32, 116, 104, 101], logprob=-0.005262688733637333, top_logprobs=[]), ChatCompletionTokenLogprob(token=' material', bytes=[32, 109, 97, 116, 101, 114, 105,...rob=-0.10020410269498825, top_logprobs=[]), ChatCompletionTokenLogprob(token=' used', bytes=[32, 117, 115, 101, 100], logprob=-1.4465793371200562, top_logprobs=[]), ChatCompletionTokenLogprob(token=' for', bytes=[32, 102, 111, 114], logprob=-0.4144509732723236, top_logprobs=[]), ChatCompletionTokenLogprob(token=' the', bytes=[32, 116, 104, 101], logprob=-0.012757238931953907, top_logprobs=[]), ChatCompletionTokenLogprob(token=' wheels', bytes=[32, 119, 104, 101, 101, 108, 115],...prob=-3.0741353034973145, top_logprobs=[]), ChatCompletionTokenLogprob(token=' on', bytes=[32, 111, 110], logprob=-0.7221264243125916, top_logprobs=[]), ChatCompletionTokenLogprob(token=' this', bytes=[32, 116, 104, 105, 115], logprob=-1.4424560070037842, top_logprobs=[]), ChatCompletionTokenLogprob(token=' truck', bytes=[32, 116, 114, 117, 99, 107], logprob=-0.6343551278114319, top_logprobs=[]), ChatCompletionTokenLogprob(token='?', bytes=[63], logprob=-0.21069929003715515, top_logprobs=[]), ChatCompletionTokenLogprob(token='<|im_end|>', bytes=[60, 124, 105, 109, 95, 101, 110...b=-4.911301948595792e-05, top_logprobs=[])]
                    special variables:
                    function variables:
                    00: ChatCompletionTokenLogprob(token='Are', bytes=[65, 114, 101], logprob=-0.007280366960912943, top_logprobs=[])
                    special variables:
                    function variables:
                    bytes: [65, 114, 101]
                    logprob: -0.007280366960912943
                    model_computed_fields: {}
                    model_config: {'extra': 'allow', 'defer_build': True}
                    model_extra: {}
                    model_fields: {'token': FieldInfo(annotation=str, required=True), 'bytes': FieldInfo(annotation=Union[List[int], NoneType], required=False, default=None), 'logprob': FieldInfo(annotation=float, required=True), 'top_logprobs': FieldInfo(annotation=List[TopLogprob], required=True)}
                    model_fields_set: {'top_logprobs', 'bytes', 'logprob', 'token'}
                    token: 'Are'
                    top_logprobs: []
                    _abc_impl: <_abc._abc_data object at 0x7a2ef7452940>
                    _calculate_keys: <bound method BaseModel._calculate_keys of ChatCompletionTokenLogprob(token='Are', bytes=[65, 114, 101], logprob=-0.007280366960912943, top_logprobs=[])>
                    _copy_and_set_values: <bound method BaseModel._copy_and_set_values of ChatCompletionTokenLogprob(token='Are', bytes=[65, 114, 101], logprob=-0.007280366960912943, top_logprobs=[])>
                    _get_value: <bound method BaseModel._get_value of <class 'openai.types.chat.chat_completion_token_logprob.ChatCompletionTokenLogprob'>>
                    _iter: <bound method BaseModel._iter of ChatCompletionTokenLogprob(token='Are', bytes=[65, 114, 101], logprob=-0.007280366960912943, top_logprobs=[])>
                    _setattr_handler: <bound method BaseModel._setattr_handler of ChatCompletionTokenLogprob(token='Are', bytes=[65, 114, 101], logprob=-0.007280366960912943, top_logprobs=[])>
                    01: ChatCompletionTokenLogprob(token=' you', bytes=[32, 121, 111, 117], logprob=-6.425174069590867e-05, top_logprobs=[])
                    02: ChatCompletionTokenLogprob(token=' asking', bytes=[32, 97, 115, 107, 105, 110, 103], logprob=-0.019521024078130722, top_logprobs=[])
                    03: ChatCompletionTokenLogprob(token=' about', bytes=[32, 97, 98, 111, 117, 116], logprob=-0.0012260308722034097, top_logprobs=[])
                    04: ChatCompletionTokenLogprob(token=' the', bytes=[32, 116, 104, 101], logprob=-0.005262688733637333, top_logprobs=[])
                    05: ChatCompletionTokenLogprob(token=' material', bytes=[32, 109, 97, 116, 101, 114, 105, 97, 108], logprob=-0.10020410269498825, top_logprobs=[])
                    06: ChatCompletionTokenLogprob(token=' used', bytes=[32, 117, 115, 101, 100], logprob=-1.4465793371200562, top_logprobs=[])
                    07: ChatCompletionTokenLogprob(token=' for', bytes=[32, 102, 111, 114], logprob=-0.4144509732723236, top_logprobs=[])
                    08: ChatCompletionTokenLogprob(token=' the', bytes=[32, 116, 104, 101], logprob=-0.012757238931953907, top_logprobs=[])
                    09: ChatCompletionTokenLogprob(token=' wheels', bytes=[32, 119, 104, 101, 101, 108, 115], logprob=-3.0741353034973145, top_logprobs=[])
                    10: ChatCompletionTokenLogprob(token=' on', bytes=[32, 111, 110], logprob=-0.7221264243125916, top_logprobs=[])
                    11: ChatCompletionTokenLogprob(token=' this', bytes=[32, 116, 104, 105, 115], logprob=-1.4424560070037842, top_logprobs=[])
                    12: ChatCompletionTokenLogprob(token=' truck', bytes=[32, 116, 114, 117, 99, 107], logprob=-0.6343551278114319, top_logprobs=[])
                    13: ChatCompletionTokenLogprob(token='?', bytes=[63], logprob=-0.21069929003715515, top_logprobs=[])
                    14: ChatCompletionTokenLogprob(token='<|im_end|>', bytes=[60, 124, 105, 109, 95, 101, 110, 100, 124, 62], logprob=-4.911301948595792e-05, top_logprobs=[])
                    len(): 15
                    """
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

                    # # Get an inference from the answer model
                    # dialog_traj = tree.get_trajectory(clarification_input_node)
                    # messages = dialog_traj.to_messages("qwen-3-vl", use_img_path=True)
                    # add_inference_messages(messages, cfg=cfg)
                    # # print(f"Testing answer model with messages:\n{messages}")

                    # inference_response_obj = await answer_model.generate(messages, use_lora=False)
                    # inference_response = inference_response_obj.choices[0].message.content
                    # assert inference_response is not None
                    # print(typer.style("\n>> Inference Response:", fg=typer.colors.BRIGHT_GREEN, bold=True))
                    # print(">> " + inference_response)
                    # tree.add_node(clarification_input_node, NodeType.INFERENCE, inference_response)

                    await _generate_inference_and_scores(tree, clarification_input_node)

                


    asyncio.run(test_servers())

    
@app.command()
def test_rl_training_datapoints(
    ctx: typer.Context,
    config_name: str = typer.Option("config", help="Name of the config file to use")
):
    from clarification_trees.models import TransformersModelV2
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name, overrides=ctx.args)

    print("Loading tree")
    test_tree_path = Path("/scratch4/home/adempst/projects/clarification-trees-v2/data/trees/tree_val_000000_11d26793-a53b-456c-add5-0d51abdea743.json")
    tree = DialogTree.load(test_tree_path / "tree.json")
    sidecar = TreeSidecar.load(test_tree_path / "tree_sidecar.json", cfg)

    print("Loading model")
    model = TransformersModelV2(cfg.clarification_model, device="cuda:7")

    print("Generating datapoints")
    datapoints, max_advantage, min_advantage = model.preprocess_rl_training_inputs(0, tree, sidecar, "user")
    print(f"Generated {len(datapoints)} datapoints.")
    for i, dp in enumerate(datapoints):
        print(f"Datapoint {i}: {dp}")

    for i, dp in enumerate(datapoints):
        print(f"\n\nDatapoint {i}")
        print(f"Datapoint string: {utils.tokens_to_str(dp.tokens, model.processor.tokenizer)}")

        # Mask out just the tokens with dp.action_mask == 1 so that we can see what part of the datapoint is the "action"
        masked_tokens: list[int] = []
        for j in range(len(dp.tokens)):
            if dp.action_mask[j] == 1:
                masked_tokens.append(dp.tokens[j])
        print(f"Masked tokens: {utils.tokens_to_str_list(masked_tokens, model.processor.tokenizer)}")
    
    print(f"Max advantage: {max_advantage}, Min advantage: {min_advantage}")
    print(f"Advantage range: {max_advantage - min_advantage}")
    



    
    
    

if __name__ == "__main__":
    # ignore_unknown_options=True allows us to pass Hydra overrides (like db.host=...)
    # without Typer throwing an "Unknown argument" error.
    app()