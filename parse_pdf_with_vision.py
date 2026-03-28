"""
Парсер PDF с поддержкой Vision LLM.

Пайплайн:
  PDF → текст (PyMuPDF) + картинки (base64) → описание картинок (Vision LLM)
      → склейка в один .txt → сохранение в parsed_docs/

Запуск:
  python parse_pdf_with_vision.py
  python parse_pdf_with_vision.py --file knowledge_base/manual.pdf
"""

import argparse
import asyncio
import base64
import logging
from pathlib import Path

import fitz  # pymupdf
from openai import AsyncOpenAI, RateLimitError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VISION_MODEL = "gemma3:12b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
KNOWLEDGE_BASE_DIR = Path("./knowledge_base")
OUTPUT_DIR = Path("./parsed_docs")

VISION_PROMPT = (
    "Это скриншот из программы 1С ERP. "
    "Опиши подробно что изображено: какое меню открыто, какие поля заполнены, "
    "какие кнопки видны, какие данные отображаются. "
    "Отвечай на русском языке, только описание — без вводных фраз."
)


def extract_page_blocks(doc: fitz.Document, page: fitz.Page, page_num: int) -> list[dict]:
    """
    Возвращает список блоков страницы в порядке их вертикальной позиции.
    Каждый блок: {"type": "text"|"image", "y": float, "content": str|bytes}
    """
    blocks = []

    # Текстовые блоки с позицией
    for block in page.get_text("blocks"):
        # block = (x0, y0, x1, y1, text, block_no, block_type)
        if block[6] == 0 and block[4].strip():  # type 0 = текст
            blocks.append({"type": "text", "y": block[1], "content": block[4].strip()})

    # Картинки с позицией
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        # Получаем bbox картинки на странице
        rects = page.get_image_rects(xref)
        y = rects[0].y0 if rects else 0
        base_image = doc.extract_image(xref)
        blocks.append({"type": "image", "y": y, "content": base_image["image"]})
        logger.debug(f"  Картинка xref={xref} на странице {page_num}, y={y:.1f}")

    # Сортируем по вертикальной позиции
    blocks.sort(key=lambda b: b["y"])
    return blocks


async def describe_image(client: AsyncOpenAI, image_bytes: bytes, label: str) -> str:
    """Отправляет картинку в Vision LLM и возвращает текстовое описание."""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    logger.info(f"  Описываю {label}...")

    for attempt in range(1, 4):
        try:
            response = await client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == 3:
                logger.error(f"  {label}: все попытки исчерпаны, пропускаю")
                return f"[{label}: не удалось получить описание — rate limit]"
            wait = attempt * 15
            logger.warning(f"  Rate limit, жду {wait} сек (попытка {attempt}/3)...")
            await asyncio.sleep(wait)
    return ""


async def process_pdf(pdf_path: Path, client: AsyncOpenAI) -> Path:
    """Обрабатывает один PDF и сохраняет результат в parsed_docs/."""
    logger.info(f"Обрабатываю: {pdf_path.name}")

    doc = fitz.open(str(pdf_path))
    result_parts = []
    img_counter = 0

    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]
        blocks = extract_page_blocks(doc, page, page_num)
        page_parts = [f"[Страница {page_num}]"]

        for block in blocks:
            if block["type"] == "text":
                page_parts.append(block["content"])
            else:
                img_counter += 1
                label = f"Изображение {img_counter} (страница {page_num})"
                desc = await describe_image(client, block["content"], label)
                page_parts.append(f"[{label}]\n{desc}")

        result_parts.append("\n".join(page_parts))

    doc.close()

    result = "\n\n".join(result_parts)
    logger.info(f"  Обработано страниц: {page_num}, картинок: {img_counter}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / (pdf_path.stem + ".txt")
    out_path.write_text(result, encoding="utf-8")
    logger.info(f"  Сохранено: {out_path} ({len(result)} символов)")

    return out_path


async def main(target_file: str | None = None):
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    if target_file:
        pdf_files = [Path(target_file)]
    else:
        pdf_files = list(KNOWLEDGE_BASE_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"PDF файлы не найдены в {KNOWLEDGE_BASE_DIR}/")
        return

    logger.info(f"Найдено PDF файлов: {len(pdf_files)}")

    for pdf_path in pdf_files:
        await process_pdf(pdf_path, client)

    logger.info("Готово! Результаты в папке parsed_docs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Парсер PDF с Vision LLM")
    parser.add_argument("--file", help="Обработать конкретный PDF файл", default=None)
    args = parser.parse_args()

    asyncio.run(main(args.file))
