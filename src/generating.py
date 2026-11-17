"""augmenting the minimalsources, loading the model and inferencing"""


from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from src.validating import MinimalSource


def read_source(source: MinimalSource) -> str:
    """reads the chunk included in the minimalsource slices of the chunks"""
    if not Path(source.file_path).is_file():
        return ""
    try:
        with open(source.file_path,
                  "r",
                  encoding="utf-8",
                  errors="replace") as f:
            content = f.read()
            return (content[
                source.first_character_index:source.last_character_index])
    except OSError:
        print("Could not open file")
        return ""


def build_context(sources: list[MinimalSource], max_chars: int = 5000) -> str:
    """building the prompt that will be given to the model"""
    context: list[str] = []
    total_chars = 0
    for i, s in enumerate(sources, start=1):
        piece = read_source(s)
        header = f"[Source {i}: {s.file_path}]"
        if total_chars + len(piece) + len(header) >= max_chars:
            break
        if not piece.strip():
            continue
        context.append(header)
        context.append(piece)
        total_chars += len(piece) + len(header) + 4
    return "\n\n".join(context)


def load_model(model_name: str = "Qwen/Qwen3-0.6B") -> tuple[Any, Any]:
    """"""
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # setting device in case gpu or cpu are avail, and the floats to handle
    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    # loading weights
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    # moving weights to device
    model.to(device)
    # what is this doing
    model.eval()
    return (tokenizer, model)


def generate_answer(question: str,
        sources: list[MinimalSource],
        tokenizer: Any,
        model: Any
        ) -> str:
    """"""
    if not question.strip():
        return ""
    context = build_context(sources)
    if not context.strip():
        return ""
    messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": context
                },
            {
                "role": "user",
                "content": f"Ctx:\n{context}\n\nQtn: {question}\n\nAns:"
                },
            ]
    # single prompt string, disable <think> blocks
    prompt: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            )
    # prompt to pytorch tensors
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # do_sample keeps answers grounded, greedy decoding
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_len:]
    answer: str = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
            ).strip()
    return answer
