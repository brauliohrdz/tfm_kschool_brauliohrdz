import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)

    def ask(self, prompt) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_ids = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs_ids = self._model.generate(**inputs_ids, max_new_tokens=200, do_sample=False)
        prompt_len = inputs_ids["input_ids"].shape[-1]
        gen_ids = outputs_ids[0, prompt_len:]

        response = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return response
