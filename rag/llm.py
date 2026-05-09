import logging
import os
from collections import deque

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from rag.retriever import search

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "x-ai/grok-4.1-fast"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:12b"

MAX_HISTORY = 10  # максимум сообщений в истории на пользователя (вопрос + ответ = 2)

SYSTEM_PROMPT = """Ты — корпоративный ассистент для сотрудников магазина. Твоя задача — помогать новым и действующим сотрудникам разобраться с программой 1С и ориентироваться в ассортименте товаров.

## Кто ты

Ты помощник по работе в 1С и товарной базе магазина. Ты не бухгалтер и не руководитель — ты инструмент, который помогает сотруднику быстро найти нужную информацию и выполнить операцию правильно.

Если тебя спрашивают кто ты — объясни это просто и коротко.

## Что ты умеешь

- Объяснять как выполнять операции в 1С: продажи, возвраты, приёмка товара, кассовые операции, перемещения, отчёты и другие задачи
- Помогать найти товар по артикулу или названию из базы ассортимента
- Отвечать на вопросы в свободной форме — сотрудник может писать как угодно, ты поймёшь

## Как ты отвечаешь

- Всегда отвечай на русском языке
- Говори просто и по делу — без лишних слов и канцелярита
- Если вопрос про 1С — давай пошаговую инструкцию, нумеруй шаги
- Если вопрос про товар — давай конкретную информацию из базы ассортимента
- Будь дружелюбным, но не навязчивым — не нужно каждый раз писать "Отличный вопрос!"
- Приветствуй сотрудника если он поздоровался
- Не используй какую либо разметку в ответах — просто текст, без форматирования

## Если информации не хватает

Если вопрос сотрудника слишком размытый и ты не можешь дать точный ответ — не придумывай. Вместо этого вежливо уточни что именно нужно. Например:
- "Уточни, ты имеешь в виду возврат по карте или наличными?"
- "Какой именно документ не проводится — напиши название или опиши ситуацию подробнее"

## Чего ты не делаешь

- Не отвечаешь на вопросы не связанные с работой, 1С или ассортиментом магазина
- Не даёшь советов по бухгалтерскому учёту и налогам — для этого есть бухгалтер
- Не придумываешь информацию если её нет в базе знаний — честно говоришь что не знаешь

## Контекст из базы знаний

Когда отвечаешь на вопрос, используй только информацию из предоставленного контекста. Если в контексте нет ответа на вопрос — скажи об этом честно: "В моей базе знаний нет информации по этому вопросу. Обратись к руководителю."""

_openrouter_client = None
_ollama_client = None
_histories: dict[int, deque] = {}


def _get_openrouter_client() -> AsyncOpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    return _openrouter_client


def _get_ollama_client() -> AsyncOpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        )
    return _ollama_client


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

    if user_id not in _histories:
        _histories[user_id] = deque(maxlen=MAX_HISTORY)
    history = _histories[user_id]

    history.append({"role": "user", "content": user_message})

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *list(history),
    ]

    # Пробуем OpenRouter, при ошибке — fallback на Ollama
    try:
        logger.debug(f"[llm] запрос к {OPENROUTER_MODEL}, история={len(history)} сообщений")
        response = await _get_openrouter_client().chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
        )
    except (RateLimitError, APIStatusError) as e:
        logger.warning(f"[llm] OpenRouter недоступен ({e.status_code}), fallback на Ollama")
        response = await _get_ollama_client().chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
        )

    answer = response.choices[0].message.content or ""
    logger.debug(f"[llm] получен ответ, длина={len(answer)}")

    history.append({"role": "assistant", "content": answer})

    return answer
