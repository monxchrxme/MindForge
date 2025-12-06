#TODO Добавить factcheck для кода
import logging
from typing import List, Dict
from services.gigachat_client import GigaChatClient

logger = logging.getLogger(__name__)


class FactCheckAgent:
    def __init__(self, client: GigaChatClient):
        self.client = client

    def verify_concepts(self, concepts: list) -> tuple[list, list]:
        """
        Возвращает кортеж: (verified_concepts, corrections_report)
        """
        if not concepts:
            return [], []

        try:
            prompt = self._build_prompt(concepts)
            response_data = self.client.generate_json(prompt)
            verified_raw = response_data.get("concepts", [])

            verified_concepts = []
            corrections_report = []

            original_map = {c.get("term"): c for c in concepts}

            for v_concept in verified_raw:
                term = v_concept.get("term")
                original = original_map.get(term)

                if original:
                    new_def = v_concept.get("definition")
                    old_def = original.get("definition")

                    # Логика обработки кода
                    code = original.get("code_snippet")
                    code_error = False

                    # Проверяем маркер ошибки кода (если вы добавили это в промпт)

                    if "[CODE_ERROR]" in new_def:
                        new_def = new_def.replace("[CODE_ERROR]", "").strip()  # 1. Чистим текст локально
                        code = None  # 2. Удаляем код локально
                        code_error = True

                    # --- ДЕТЕКЦИЯ ИЗМЕНЕНИЙ ---

                    # 1. Определение (игнорируем мелкие пробелы/регистр в начале)
                    change_type = v_concept.get("change_type",
                                                "major")  # если модель не указала — считаем важной правкой

                    # --- ДЕТЕКЦИЯ ИЗМЕНЕНИЙ ---

                    if change_type == "major":
                        # Считаем это существенной правкой, показываем пользователю
                        corrections_report.append({
                            "term": term,
                            "type": "definition_fix",
                            "message": "Существенная правка определения.",
                            "original": old_def,
                            "fixed": new_def
                        })
                    elif change_type == "minor":
                        # Можем залогировать в DEBUG, но не включать в отчёт для UI
                        logger.info(f"[FactCheck] Minor tweak for term '{term}' (definition improved, смысл тот же)")
                        # Если хочешь, можешь добавить отдельный тип в отчет, но уже не как WARNING
                        # corrections_report.append({... тип: 'definition_minor' ...})
                    else:  # "none"
                        # Ничего не делаем, даже если строки не совпадают
                        pass

                    # 2. Код
                    if code_error:
                        corrections_report.append({
                            "term": term,
                            "type": "code_mismatch",
                            "message": "Код не соответствует термину и был удален.",
                            "removed_code": original.get("code_snippet")
                        })
                    # --------------------------

                    verified_concepts.append({
                        "term": term,
                        "definition": new_def,
                        "code_snippet": code
                    })

            # Лог внутри агента (краткий)
            logger.info(f"Verified {len(verified_concepts)} concepts, found {len(corrections_report)} diffs")

            return verified_concepts, corrections_report

        except Exception as e:
            logger.error(f"FactCheck error: {e}")
            # При ошибке возвращаем оригиналы и пустой отчет
            return concepts, []

    def _build_prompt(self, concepts: List[Dict[str, str]]) -> str:
        concepts_list = ""
        for i, concept in enumerate(concepts, 1):
            term = concept.get('term', '')
            definition = concept.get('definition', '')
            code = concept.get('code_snippet')

            # Экранирование для f-string
            term_safe = str(term).replace("{", "{{").replace("}", "}}")
            def_safe = str(definition).replace("{", "{{").replace("}", "}}")

            concepts_list += f"--- КОНЦЕПТ {i} ---\n"
            concepts_list += f"Термин: {term_safe}\n"
            concepts_list += f"Определение: {def_safe}\n"

            if code:
                # Экранируем код для контекста
                code_safe = str(code).replace("{", "{{").replace("}", "}}")
                concepts_list += f"Код (для контекста): \n{code_safe}\n"

            concepts_list += "\n"

        prompt = f"""
Твоя роль: Строгий научный редактор и программист.
Твоя задача: Проверить список концептов на корректность и согласованность.

ВХОДНЫЕ ДАННЫЕ:
{concepts_list}

ИНСТРУКЦИИ:
1. **Фактическая точность:** Проверь определение. Если есть ошибка — исправь.
2. **Связь с кодом:** Если показан код, проверь, соответствует ли определение этому коду.
   - Если определение противоречит коду -> ИСПРАВЬ определение.
   - Если код содержит грубые ошибки -> добавь в начало определения пометку "[CODE_ERROR]".
3. **Классификация изменений:** Оцени серьезность своей правки.

ФОРМАТ ОТВЕТА (JSON):
{{
  "concepts": [
    {{
      "term": "Термин (не меняй!)",
      "definition": "Исправленный текст",
      "change_type": "..." // "none", "minor" или "major"
    }}
  ]
}}

ЗНАЧЕНИЯ change_type:
- "none": Текст не менялся.
- "minor": Косметические правки (стиль, уточнение), смысл не изменился.
- "major": Исправление фактической ошибки, изменение смысла (напр. "удалить" -> "добавить"), или обнаружено несоответствие коду.

ВАЖНО: 
- Верни ТОЛЬКО JSON. 
- НЕ возвращай поле 'code_snippet' в JSON.
"""
        return prompt

