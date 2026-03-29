import logging

from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from bot.keyboards import get_main_keyboard
from rag.llm import ask, clear_history

logger = logging.getLogger(__name__)
router = Router()

WELCOME_TEXT = (
    "Привет! Я — 1С Ассистент.\n\n"
    "Помогаю новым сотрудникам разобраться в работе с 1С ERP.\n\n"
    "Просто напиши свой вопрос — например:\n"
    "• «Как создать новый заказ покупателя?»\n"
    "• «Где найти отчёт по остаткам?»\n"
    "• «Как провести инвентаризацию?»\n\n"
    "Я найду ответ в базе знаний компании."
)

HELP_TEXT = (
    "Я отвечаю на вопросы по работе в 1С ERP.\n\n"
    "Просто напишите вопрос в свободной форме.\n"
    "Поиск ведётся по базе знаний компании."
)


@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"[/start] user_id={message.from_user.id} username={message.from_user.username}")
    await message.answer(WELCOME_TEXT, reply_markup=get_main_keyboard())


@router.message(Command("help"))
async def cmd_help(message: Message):
    logger.info(f"[/help] user_id={message.from_user.id}")
    await message.answer(HELP_TEXT)


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    clear_history(message.from_user.id)
    logger.info(f"[/clear] user_id={message.from_user.id} история очищена")
    await message.answer("История диалога очищена.")


@router.message()
async def handle_question(message: Message):
    if not message.text:
        return

    if message.text.strip() == "Помощь":
        logger.info(f"[Помощь] user_id={message.from_user.id}")
        await message.answer(HELP_TEXT)
        return

    logger.info(f"[question] user_id={message.from_user.id} text={message.text!r}")

    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action=ChatAction.TYPING,
    )

    try:
        answer = await ask(message.text, message.from_user.id)
        logger.info(f"[ответ] user_id={message.from_user.id} answer_len={len(answer)}")
        await message.answer(answer)
    except Exception as e:
        logger.error(f"[ошибка LLM] user_id={message.from_user.id} error={e}", exc_info=True)
        await message.answer(
            "Извините, произошла ошибка при обработке вашего запроса. "
            "Попробуйте ещё раз через несколько секунд."
        )
