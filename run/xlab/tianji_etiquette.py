"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers.
We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
Please run with the command `streamlit run path/to/web_demo.py --server.address=0.0.0.0 --server.port 7860`.
Using `python path/to/web_demo.py` may cause unknown problems.
"""
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
import os
import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging

from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip
from modelscope import snapshot_download

logger = logging.get_logger(__name__)

# 提前下载模型
model_path = "./huggingface/internlm/internlm2-7b"
# os.system(f'git clone https://code.openxlab.org.cn/sanbuphy/tianji-etiquette-internlm2-7b.git {model_path}')
# os.system(f'cd {model_path} && git lfs pull')

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    """
    使用给定模型和分词器交互式地生成文本。该函数通过逐步解码的方式生成输出，并在每次生成新token后立即返回当前结果，
    支持流式响应。

    参数:
        model: 预训练的语言模型。
        tokenizer: 对应于模型的分词器。
        prompt (str): 输入提示文本，用于引导生成过程。
        generation_config (Optional[GenerationConfig]): 控制生成行为的配置对象（如最大长度、采样策略等）。
        logits_processor (Optional[LogitsProcessorList]): 用于处理logits的处理器列表。
        stopping_criteria (Optional[StoppingCriteriaList]): 判断是否停止生成的标准列表。
        prefix_allowed_tokens_fn (Optional[Callable[[int, torch.Tensor], List[int]]]):
            允许前缀tokens的函数，用于限制生成词汇。
        additional_eos_token_id (Optional[int]): 额外指定的结束符ID，在检测到时也会终止生成。
        **kwargs: 其他传递给generation_config的参数。

    返回:
        Generator[str]: 每次yield一个字符串形式的部分生成结果。
    """
    # 判断是使用cpu还是gpu
    useDevice = "cuda" if torch.cuda.is_available() else "cpu"

    # 将输入提示文本进行编码并移至GPU设备
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    for k, v in inputs.items():
        inputs[k] = v.to(useDevice)
        # inputs[k] = v.cuda()
    input_ids = inputs["input_ids"]
    batch_size, input_ids_seq_length = (
        input_ids.shape[0],
        input_ids.shape[-1],
    )  # noqa: F841  # pylint: disable=W0612

    # 若未提供generation_config则使用模型默认配置，并更新相关参数
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)

    # 获取开始与结束标记ID，并根据需要扩展结束标记集合
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)

    # 处理max_length和max_new_tokens之间的兼容性警告
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using [max_length](file://d:\workspace\vmshareroom\python_project\TianJi\tianji\finetune\xtuner\internlm2_chat_7b_full_finetune.py#L49-L49)'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = (
            generation_config.max_new_tokens + input_ids_seq_length
        )
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and [max_length](file://d:\workspace\vmshareroom\python_project\TianJi\tianji\finetune\xtuner\internlm2_chat_7b_full_finetune.py#L49-L49)(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    # 警告：如果输入序列已经超过了设定的最大长度
    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but [max_length](file://d:\workspace\vmshareroom\python_project\TianJi\tianji\finetune\xtuner\internlm2_chat_7b_full_finetune.py#L49-L49) is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 初始化logits处理器和停止条件判断器
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    # 初始化未完成序列状态及得分记录变量
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None

    # 循环生成下一个token直到满足停止条件
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # 前向传播获取下一token的概率分布
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # 应用logits处理器和warper调整概率分布
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # 根据do_sample决定是采样还是贪婪选择最高概率token
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # 更新已生成的token ID序列以及模型输入参数
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )

        # 更新未完成序列的状态（若遇到EOS token则标记为已完成）
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long()
        )

        # 解码当前生成的结果并去除可能存在的EOS token
        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        # 实时返回部分生成结果
        yield response

        # 当所有句子都已完成或达到最大长度时退出循环
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    # 判断是使用cpu还是gpu
    useDevice = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        .to(torch.bfloat16)
        .to(useDevice).eval()
        # .cuda()

        # .to(useDevice).eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=8, max_value=32768, value=32768)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)

    generation_config = GenerationConfig(
        max_length=max_length, top_p=top_p, temperature=temperature
    )

    return generation_config


user_prompt = "<|im_start|>user\n{user}<|im_end|>\n"
robot_prompt = "<|im_start|>assistant\n{robot}<|im_end|>\n"
cur_query_prompt = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = "你现在是由SocialAI开发的人情世故大模型，你的任务是洞察人情世故、提供合适的交往策略和建议。在处理问题时，你应当考虑到文化背景、社会规范和个人情感，以帮助用户更好地理解复杂的人际关系和社会互动。"
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = user_prompt.format(user=cur_content)
        elif message["role"] == "robot":
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print("load model begin.")
    model, tokenizer = load_model()
    print("load model end.")

    st.title("人情世故大模型-敬酒礼仪文化模块")

    generation_config = prepare_generation_config()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("robot"):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                additional_eos_token_id=92542,
                **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(
                cur_response
            )  # pylint: disable=undefined-loop-variable
        # Add robot response to chat history
        st.session_state.messages.append(
            {
                "role": "robot",
                "content": cur_response,  # pylint: disable=undefined-loop-variable
            }
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
