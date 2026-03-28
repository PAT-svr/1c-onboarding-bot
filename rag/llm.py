import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from rag.retriever import search

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "arcee-ai/trinity-large-preview:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
]
MAX_RETRIES = 3
RETRY_DELAY = 10  # секунд между попытками

SYSTEM_PROMPT = """Ты — корпоративный ассистент по работе с 1С ERP. Помогаешь новым сотрудникам разобраться в системе.

Правила:
- Отвечай на русском языке, кратко и по делу.
- На приветствия и общие фразы ("привет", "спасибо", "пока") отвечай естественно и коротко, без упоминания 1С.
- Если вопрос не связан с 1С — вежливо сообщи, что ты специализируешься только на 1С ERP.
- Если вопрос по 1С — давай конкретный ответ, при наличии шагов оформляй нумерованным списком. Не используй выдиление жирним или курсивом.
- Не придумывай информацию, которой нет в базе знаний. Если не знаешь — скажи, что не знаешь, и предложи обратиться к специалисту."""

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    return _client


async def ask(user_question: str) -> str:
    context_chunks = search(user_question, k=3)
    logger.debug(f"[retriever] найдено фрагментов: {len(context_chunks)}")

    if context_chunks:
        context_text = "\n\n---\n\n".join(context_chunks)
        user_message = (
            f"Используй следующие фрагменты из базы знаний:\n\n"
            f"{context_text}\n\n"
            f"Вопрос сотрудника: {user_question}"
        )
    else:
        logger.warning("[retriever] база знаний пуста, отвечаем из общих знаний")
        user_message = (
            f"База знаний пока не загружена. "
            f"Ответь на вопрос из общих знаний о 1С ERP.\n\n"
            f"Вопрос: {user_question}"
        )

    client = _get_client()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for model in MODELS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(f"[llm] запрос к {model} (попытка {attempt}/{MAX_RETRIES})")
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                answer = response.choices[0].message.content or ""
                logger.debug(f"[llm] получен ответ от {model}, длина={len(answer)}")
                return answer
            except (RateLimitError, APIStatusError) as e:
                logger.warning(f"[llm] ошибка {e.status_code} на {model} (попытка {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
        logger.warning(f"[llm] {model} недоступна, пробуем следующую...")

    return "Сервис временно перегружен. Попробуйте через минуту."
