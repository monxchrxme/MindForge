import logging
import ast
from typing import List, Dict

from services.gigachat_client import GigaChatClient

logger = logging.getLogger(__name__)


class FactCheckAgent:
    def __init__(self, client: GigaChatClient):
        self.client = client

    def _check_code_syntax(self, concepts: List[Dict[str, str]]) -> Dict:
        """
        Базовый локальный tool: проверяет синтаксис Python-кода в code_snippet.
        Всегда вызывается перед LLM.
        """
        results = {
            "syntax_errors": [],
            "valid_terms": []
        }

        for concept in concepts:
            term = concept.get("term", "Unknown")
            code = concept.get("code_snippet")

            if not code:
                continue

            try:
                ast.parse(code)
                results["valid_terms"].append(term)
                logger.debug(f"[SyntaxCheck] {term}: OK")
            except SyntaxError as e:
                msg = f"Line {e.lineno}: {e.msg}"
                results["syntax_errors"].append(
                    {"term": term, "error": msg, "code": code}
                )
                logger.warning(f"[SyntaxCheck] {term}: {msg}")

        return results

    def _get_ast_depth(self, node: ast.AST, level: int = 0) -> int:
        """
        Оценка глубины вложенности AST (простая эвристика сложности кода).
        """
        if not list(ast.iter_child_nodes(node)):
            return level
        return max(self._get_ast_depth(child, level + 1)
                   for child in ast.iter_child_nodes(node))

    def _find_duplicate_lines(self, lines: List[str]) -> List[str]:
        """
        Поиск дублирующихся непустых строк.
        """
        seen = {}
        duplicates = set()
        for line in lines:
            if not line:
                continue
            if line in seen:
                duplicates.add(line)
            else:
                seen[line] = True
        return list(duplicates)

    def _extract_imports(self, code: str) -> List[str]:
        """
        Упрощённый сбор имён импортов (для эвристики неиспользуемых импортов).
        """
        names = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return names

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    names.append(f"{module}.{alias.name}".strip("."))
        return names

    def _advanced_code_quality(self, concepts: List[Dict[str, str]]) -> Dict:
        """
        Продвинутый локальный tool анализа качества кода.
        Вызывается только для относительно сложного кода (много строк).
        Возвращает по каждому термину рейтинг и список проблем.
        """
        results = []

        for concept in concepts:
            term = concept.get("term", "Unknown")
            code = concept.get("code_snippet", "")

            if not code:
                continue

            lines = [l.rstrip() for l in code.splitlines()]
            nonempty_lines = [l.strip() for l in lines if l.strip()]

            # Простой код — пропускаем, чтобы tool не вызывался "всегда"
            if len(nonempty_lines) < 5:
                results.append({
                    "term": term,
                    "status": "simple",
                    "quality_score": None,
                    "issues": [],
                    "needs_review": False
                })
                continue

            score = 100
            issues: List[str] = []

            # 1. Сложность по глубине AST
            try:
                tree = ast.parse(code)
                depth = self._get_ast_depth(tree)
                if depth > 4:
                    score -= 20
                    issues.append(f"Высокая вложенность конструкций (глубина {depth})")
            except SyntaxError:
                # Сюда обычно не дойдём, синтаксис отфильтровал _check_code_syntax
                score -= 30
                issues.append("Код не парсится (SyntaxError)")

            # 2. Дубликаты строк
            duplicates = self._find_duplicate_lines(nonempty_lines)
            if duplicates:
                score -= 15
                issues.append(f"Повторяющиеся строки кода ({len(duplicates)})")

            # 3. Импорты (очень грубая эвристика)
            imports = self._extract_imports(code)
            if len(imports) >= 3 and len(set(imports)) < len(imports):
                score -= 10
                issues.append("Возможны неиспользуемые или дублирующиеся импорты")

            score = max(0, score)
            needs_review = score < 80 or bool(issues)

            results.append({
                "term": term,
                "status": "analyzed",
                "quality_score": score,
                "issues": issues,
                "needs_review": needs_review
            })

            logger.info(
                f"[QualityCheck] {term}: score={score}, "
                f"issues={len(issues)}, needs_review={needs_review}"
            )

        has_critical = any(r.get("needs_review") for r in results)
        return {"results": results, "critical_issues": has_critical}

    def verify_concepts(self, concepts: list) -> tuple[list, list]:
        """
        Основной метод фактчека.
        Возвращает (verified_concepts, corrections_report).
        """
        if not concepts:
            return [], []

        try:
            corrections_report: List[Dict] = []

            # ЭТАП 1. Всегда: базовая проверка синтаксиса
            syntax_check = self._check_code_syntax(concepts)
            for err in syntax_check["syntax_errors"]:
                corrections_report.append({
                    "term": err["term"],
                    "type": "syntax_error",
                    "message": err["error"],
                    "code": err["code"]
                })

            # ЭТАП 2. Опционально: продвинутый анализ качества кода
            has_potentially_complex = any(
                c.get("code_snippet") and
                len([l for l in str(c.get("code_snippet")).splitlines()
                     if l.strip()]) >= 5
                for c in concepts
            )
            if has_potentially_complex:
                quality = self._advanced_code_quality(concepts)
                for item in quality["results"]:
                    if item.get("needs_review"):
                        msg_parts = []
                        score = item.get("quality_score")
                        if score is not None:
                            msg_parts.append(f"оценка качества кода {score} из 100")
                        if item.get("issues"):
                            msg_parts.append("; ".join(item["issues"]))
                        message = ". ".join(msg_parts) if msg_parts else "Код требует ручной проверки"
                        corrections_report.append({
                            "term": item["term"],
                            "type": "code_quality",
                            "message": message
                        })

            # ЭТАП 3. LLM-фактчек определений и связи с кодом
            prompt = self._build_prompt(concepts)
            response_data = self.client.generate_json(prompt)

            verified_raw = response_data.get("concepts", [])
            verified_concepts: List[Dict] = []

            original_map = {c.get("term"): c for c in concepts}

            for v_concept in verified_raw:
                term = v_concept.get("term")
                original = original_map.get(term)

                if not original:
                    continue

                new_def = v_concept.get("definition")
                old_def = original.get("definition")
                code = original.get("code_snippet")
                code_error = False

                if isinstance(new_def, str) and "[CODE_ERROR]" in new_def:
                    new_def = new_def.replace("[CODE_ERROR]", "").strip()
                    code = None
                    code_error = True

                change_type = v_concept.get("change_type", "major")

                if change_type == "major":
                    corrections_report.append({
                        "term": term,
                        "type": "definition_fix",
                        "message": "Существенная правка определения.",
                        "original": old_def,
                        "fixed": new_def
                    })
                elif change_type == "minor":
                    logger.info(f"[FactCheck] Minor tweak for term '{term}'")

                if code_error:
                    corrections_report.append({
                        "term": term,
                        "type": "code_mismatch",
                        "message": "Код не соответствует термину и был удалён.",
                        "removed_code": original.get("code_snippet")
                    })

                verified_concepts.append({
                    "term": term,
                    "definition": new_def,
                    "code_snippet": code
                })

            logger.info(
                f"Verified {len(verified_concepts)} concepts, "
                f"found {len(corrections_report)} issues"
            )
            return verified_concepts, corrections_report

        except Exception as e:
            logger.error(f"FactCheck error: {e}", exc_info=True)
            # На всякий случай возвращаем исходные концепты и то, что успели собрать
            return concepts, corrections_report if 'corrections_report' in locals() else []

    def _build_prompt(self, concepts: List[Dict[str, str]]) -> str:
        concepts_list = ""
        for i, concept in enumerate(concepts, 1):
            term = concept.get('term', '')
            definition = concept.get('definition', '')
            code = concept.get('code_snippet')

            term_safe = str(term).replace("{", "{{").replace("}", "}}")
            def_safe = str(definition).replace("{", "{{").replace("}", "}}")

            concepts_list += f"--- КОНЦЕПТ {i} ---\n"
            concepts_list += f"Термин: {term_safe}\n"
            concepts_list += f"Определение: {def_safe}\n"

            if code:
                code_safe = str(code).replace("{", "{{").replace("}", "}}")
                concepts_list += f"Код (для контекста): \n{code_safe}\n"

            concepts_list += "\n"

        prompt = f"""
Твоя роль: строгий научный редактор и программист.

Твоя задача: проверить список концептов на корректность и согласованность.

ВХОДНЫЕ ДАННЫЕ:

{concepts_list}

ИНСТРУКЦИИ:

1. Фактическая точность: проверь определение. Если есть ошибка — исправь.
2. Связь с кодом: если показан код, проверь, соответствует ли определение этому коду.
   - Если определение противоречит коду — исправь определение.
   - Если код содержит грубые ошибки — добавь в начало определения пометку "[CODE_ERROR]".
3. Классификация изменений: оцени серьёзность своей правки.

ФОРМАТ ОТВЕТА (JSON):

{{
    "concepts": [
        {{
            "term": "Термин (не меняй!)",
            "definition": "Исправленный текст",
            "change_type": "none|minor|major"
        }}
    ]
}}

ЗНАЧЕНИЯ change_type:
- "none": текст не менялся;
- "minor": косметические правки, смысл не изменился;
- "major": исправление фактической ошибки или несоответствия коду.

ВАЖНО:
- Верни только JSON.
- Не возвращай поле "code_snippet" в JSON.
"""
        return prompt
