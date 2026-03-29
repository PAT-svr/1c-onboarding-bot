import logging
from collections import deque

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from rag.retriever import search

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "gemma3:12b"
MAX_HISTORY = 10  # максимум сообщений в истории на пользователя (вопрос + ответ = 2)

SYSTEM_PROMPT = """Ты — корпоративный ассистент по работе с 1С ERP. Помогаешь новым сотрудникам разобраться в системе.

Правила:
- Отвечай на русском языке, кратко и по делу.
- На приветствия и общие фразы ("привет", "спасибо", "пока") отвечай естественно и коротко, без упоминания 1С.
- Если вопрос не связан с 1С — вежливо сообщи, что ты специализируешься только на 1С ERP.
- Если вопрос по 1С — давай конкретный ответ, при наличии шагов оформляй нумерованным списком. Не используй выдиление жирним или курсивом.
- Не придумывай информацию, которой нет в базе знаний. Если не знаешь — скажи, что не знаешь, и предложи обратиться к специалисту."""

_client = None
# Словарь: user_id -> deque сообщений (role, content)
_histories: dict[int, deque] = {}


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        )
    return _client


def clear_history(user_id: int) -> None:
    """Очищает историю диалога для пользователя."""
    _histories.pop(user_id, None)


async def ask(user_question: str, user_id: int) -> str:
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

    # Получаем или создаём историю для пользователя
    if user_id not in _histories:
        _histories[user_id] = deque(maxlen=MAX_HISTORY)
    history = _histories[user_id]

    # Добавляем новый вопрос в историю
    history.append({"role": "user", "content": user_message})

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *list(history),
    ]

    client = _get_client()
    logger.debug(f"[llm] запрос к {MODEL}, история={len(history)} сообщений")
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    answer = response.choices[0].message.content or ""
    logger.debug(f"[llm] получен ответ, длина={len(answer)}")

    # Сохраняем ответ в историю
    history.append({"role": "assistant", "content": answer})

    return answer
