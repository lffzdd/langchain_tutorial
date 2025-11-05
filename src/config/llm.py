from langchain_openai.chat_models import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek

vllm = ChatOpenAI(
    model="Qwen/Qwen3-14B-AWQ",
    base_url="http://localhost:8600/v1",
    api_key="EMPTY",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

deepseek = ChatDeepSeek(
    model="deepseek-chat", api_key="sk-4d60ba5196d14f939126d5e3b5f1647a"
)
