"""augmenting the minimalsources, loading the model and inferencing"""


from typing import Any
from tqdm import tqdm
from pathlib import Path
from src.validating import MinimalSource


def read_source(source: MinimalSource) -> str:
    """reads the chunk included in the minimalsource slices of the chunks"""
    if not Path(source.file_path).is_file():
        return ""
    try:
        with open(
                source.file_path,
                "r",
                encoding="utf-8",
                ) as f:
            content = f.read()
            # slice the file using the character offsets in the chunk
            return content[
                source.first_character_index:source.last_character_index
                ]
    except Exception as e:
        print(e)
        return ""


def build_context(sources: list[MinimalSource], max_chars: int = 2000) -> str:
    """building the prompt that will be given to the model"""
    # formatted [source 0: path] header and text
    context: list[str] = []
    # tracks total chars needed so always below max_chars
    total_chars = 0
    for i, s in enumerate(sources, start=1):
        piece = read_source(s)
        # skips empty chunks
        if not piece.strip():
            continue
        header = f"[Source {i}: {s.file_path}]"
        # calc char budget knowing header + \n\n separators (4 chars)
        remaining = max_chars - total_chars - len(header) - 4
        if remaining <= 0:
            break
        # if chunk doesnt fit trim it if context empty, else stop adding
        if len(piece) > remaining:
            if not context:
                piece = piece[:remaining]
            else:
                break
        context.append(header)
        context.append(piece)
        total_chars += len(piece) + len(header) + 4
    # join headers and snippets into a context block
    return "\n\n".join(context)


def load_model(model_name: str = "Qwen/Qwen3-0.6B") -> tuple[Any, Any]:
    """loads tokenizer and causal LM on device"""
    try:
        with tqdm(total=2, desc="Loading ML libs") as pbar:
            import torch
            pbar.update(1)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            pbar.update(1)
        # tokenizer converts strings into intes token IDs and back
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # left padding aligns the end of prompts for casual text gen
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # loading causal weights of qwen
        model: Any = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                )
        # put weights in CPU ram
        model.to(torch.device("cpu"))
        # switches model from training to evaluation / inference
        # disables training layers
        model.eval()
        return (tokenizer, model)
    except Exception as e:
        raise RuntimeError("Model failed to load") from e


def generate_answer(
        question: str,
        sources: list[MinimalSource],
        tokenizer: Any,
        model: Any,
        ) -> str:
    """generates answer from sources"""
    if not question.strip():
        return ""
    if tokenizer is None or model is None:
        return ""
    # build 2000 char context from retrieved chunks
    context = build_context(sources)
    if not context.strip():
        return ""
    import torch
    from transformers.generation.streamers import BaseStreamer

    # BaseStreamer gets callbacks from model.generate() every step
    class _ProgressStreamer(BaseStreamer):
        """updates a tqdm bar as tokens are generated"""
        def __init__(self, tokens: int = 512) -> None:
            """initializes the prompt prefill progress bar"""
            self.max_tokens = tokens
            # first put() passes the input prompt, not a generated token
            self.is_prompt = True
            self.gen_pbar: Any = None

        def put(self, value: Any) -> None:
            """finishes prefill bar and ticks new token generation bar"""
            if self.is_prompt:
                self.is_prompt = False
                return
            if self.gen_pbar is None:
                self.gen_pbar = tqdm(
                    total=self.max_tokens,
                    desc="Generating answer",
                )
            if self.gen_pbar.n < self.max_tokens:
                self.gen_pbar.update(1)

        def end(self) -> None:
            """snaps active bars to 100% and closes them at early EOS"""
            if self.gen_pbar is not None:
                if self.gen_pbar.n > 0:
                    self.gen_pbar.total = self.gen_pbar.n
                else:
                    self.gen_pbar.update(self.gen_pbar.total)
                self.gen_pbar.refresh()
                self.gen_pbar.close()

    # chat roles structure and retrieved context for qwen
    # structure instructions + retrieved context + questions
    messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "Answer concisely using the provided context"
                },
            {
                "role": "user",
                "content": f"Ctx:\n{context}\n\nQtn: {question}\n\nAns:"
                },
            ]
    try:
        # format into Qwen prompt first step, disable <think> blocks
        # tokenize false so HF doesnt turn the prompt in numbers yet
        # addgenerationprompt adds an answer block at the end
        prompt: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                )
        # turning prompt into pytorch tensors pt
        # return tensors grid pt will be input into model generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # attach hooks to 28 decoder layers to track prefill
        layers = getattr(getattr(model, "model", None), "layers", [])
        total_layers = max(len(layers), 1)
        prefill_pbar = tqdm(
                total=total_layers,
                desc="Prefilling prompt",
                )
        handles: list[Any] = []

        def _prefill_hook(module: Any, args: Any, output: Any) -> None:
            """ticks bar per transformer layer"""
            if prefill_pbar.n < total_layers:
                prefill_pbar.update(1)
                if prefill_pbar.n == total_layers:
                    prefill_pbar.close()
        for layer in layers:
            handles.append(layer.register_forward_hook(_prefill_hook))
        # do_sample keeps answers grounded, greedy decoding
        try:
            # disable autograd to save RAM do_sample stops creativity
            # progress streamers inherits from basestreamer which is a
            # transformers hook activated every time a token is produced
            with torch.no_grad():
                outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        streamer=_ProgressStreamer(tokens=512)
                    )
        finally:
            # always removing hooks in future question calls
            for h in handles:
                h.remove()
            if not prefill_pbar.disable:
                prefill_pbar.close()
        # outputs[0] contains [prompttokens + answertokens]
        prompt_len = len(inputs["input_ids"][0])
        new_tokens = outputs[0][prompt_len:]
        # convert generated token ids back to text without special tags
        answer: str = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
                ).strip()
        return answer
    except Exception as e:
        print(e)
        return ""
