import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


class GeminiMedicalLLM:
    """
    HuggingFace API LLM (NO DOWNLOAD MODEL)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    ):

        api_key = os.getenv("HF_TOKEN")

        if not api_key:
            raise ValueError("HF_TOKEN not found in .env")

        self.client = InferenceClient(
            model=model_name,
            token=api_key
        )

        print(f"[INFO] Using HF API model: {model_name}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:

        response = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
        )

        # aman untuk berbagai format HF
        return response.choices[0].message.content

    def debug_generate(self, prompt: str):

        print("\n========== HF API RESPONSE ==========\n")

        print(self.generate(prompt))