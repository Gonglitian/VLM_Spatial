from vllm import LLM, SamplingParams
from PIL import Image

img = Image.open("../data/image.png")

llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True          # Qwen custom tokenizer
)

# 使用正确的聊天模板格式
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "请用一句话描述这张图片。"}
        ]
    }
]

# 应用聊天模板
prompt = llm.get_tokenizer().apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": img}
    },
    sampling_params=SamplingParams(temperature=0.2)
)

print(outputs[0].outputs[0].text)
