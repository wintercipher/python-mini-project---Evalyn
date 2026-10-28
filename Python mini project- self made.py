import argparse
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from pypdf import PdfReader
from pypdf.errors import PdfReadError


APP_DIRECTORY = Path(__file__).resolve().parent
app = Flask(__name__)

QUESTION_START = re.compile(
    r"(?im)^\s*(?:Q(?:uestion)?\s*)?(\d+)(?=\s*(?:[.):\]-]|\[))"
)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}
OBJECTIVE_TYPES = {"mcq", "tf", "blank", "numeric"}
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
NUMERIC_ANSWER_PATTERN = re.compile(
    r"^\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?(?:\s*[A-Za-z%°²³/^.-][A-Za-z0-9%°²³/^.-]*)*\s*$"
)


def extract_pdf_text(pdf_source, source_name: str = "PDF") -> str:
    """Extract selectable typed text; scanned image pages are not supported."""
    reader = PdfReader(pdf_source)
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    unreadable_pages = [
        str(index + 1) for index, text in enumerate(page_text) if not text
    ]
    if unreadable_pages:
        raise ValueError(
            f"{source_name!r} has no selectable text on page(s) "
            f"{', '.join(unreadable_pages)}. Scanned or handwritten pages "
            "aren't supported; use a typed PDF with selectable text."
        )

    text = "\n".join(page_text).strip()
    if not text:
        raise ValueError(
            f"No selectable text was found in {source_name!r}. "
            "Use a typed PDF with selectable text."
        )
    return text


def split_questions(text: str) -> dict[int, str]:
    matches = list(QUESTION_START.finditer(text))
    return {
        int(match.group(1)): re.sub(
            r"^[.):\]-]\s*", "",
            text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip(),
        )
        for i, match in enumerate(matches)
    }


def infer_question_type(
    question_text: str,
    answer: str,
    marks: float,
    has_rubric: bool,
    question_block: str,
) -> str:
    answer_normalized = normalize(answer)
    if answer_normalized in {"true", "false", "yes", "no"}:
        return "tf"
    if re.fullmatch(r"[a-d]", answer_normalized):
        return "mcq"
    if NUMERIC_ANSWER_PATTERN.fullmatch(answer):
        return "numeric"
    if re.search(
        r"_{2,}|(?:fill|complete)\s+in\s+(?:the\s+)?blank",
        question_text,
        re.I,
    ):
        return "blank"
    if re.search(r"(?im)^\s*(?:\([A-D]\)|[A-D][.)])\s+\S", question_block):
        return "mcq"
    if has_rubric or marks >= 8:
        return "long"
    return "short"


def parse_answer_key(text: str) -> dict[int, dict[str, object]]:
    questions = {}
    for number, block in split_questions(text).items():
        marks_match = re.search(r"\[(\d+(?:\.\d+)?)\s*marks?\]", block, re.I)
        type_match = re.search(r"\[type:\s*(\w+)\s*\]", block, re.I)
        answer_match = re.search(r"(?ims)^\s*Answer:\s*(.*?)(?=^\s*(?:Keywords:|Rubric:)|\Z)", block)
        if not answer_match:
            continue

        answer = answer_match.group(1).strip()
        marks = float(marks_match.group(1)) if marks_match else 1.0
        question_text = re.sub(
            r"(?im)^\s*Question:\s*|\[(?:\d+(?:\.\d+)?\s*marks?|type:\s*\w+)\]",
            "", block[:answer_match.start()],
        )
        keywords = re.search(r"(?im)^\s*Keywords:\s*(.*)$", block)
        questions[number] = {
            "answer": answer,
            "marks": marks,
            "type": type_match.group(1).lower() if type_match else infer_question_type(
                question_text, answer, marks,
                bool(re.search(r"(?im)^\s*Rubric:", block)), block,
            ),
            "keywords": [
                word.strip()
                for word in re.split(r"[,;]", keywords.group(1))
                if word.strip()
            ] if keywords else [],
        }
    return questions


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def keywords_from(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"\w+", text.casefold())
        if len(word) > 1 and word not in STOP_WORDS
    }


def first_number(text: str) -> Decimal | None:
    for match in NUMBER_PATTERN.finditer(text):
        token = match.group().replace(",", "")
        if token in {"", "+", "-", "."}:
            continue
        try:
            value = Decimal(token)
        except InvalidOperation:
            continue
        if value.is_finite():
            return value
    return None


def one_edit_apart(first: str, second: str) -> bool:
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) > len(second):
        first, second = second, first

    first_index = second_index = edits = 0
    while first_index < len(first) and second_index < len(second):
        if first[first_index] == second[second_index]:
            first_index += 1
            second_index += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(first) == len(second):
            first_index += 1
        second_index += 1

    if second_index < len(second) or first_index < len(first):
        edits += 1
    return edits == 1


def grade_answer(expected: dict[str, object], student_answer: str) -> float:
    expected_answer = str(expected["answer"])
    question_type = str(expected["type"])
    maximum = float(expected["marks"])

    if question_type == "numeric":
        expected_number = first_number(expected_answer)
        student_number = first_number(student_answer)
        if expected_number is None or student_number is None:
            return 0.0
        rounding_tolerance = Decimal("0.5").scaleb(
            expected_number.as_tuple().exponent
        )
        return (
            maximum
            if abs(student_number - expected_number) <= rounding_tolerance
            else 0.0
        )

    if question_type in OBJECTIVE_TYPES:
        normalized_expected = normalize(expected_answer)
        normalized_student = normalize(student_answer)
        matches = normalized_student == normalized_expected
        if (
            question_type == "blank"
            and not matches
            and len(normalized_expected) >= 5
            and " " not in normalized_expected
            and " " not in normalized_student
        ):
            matches = one_edit_apart(normalized_expected, normalized_student)
        return maximum if matches else 0.0

    keywords = set()
    for keyword in expected["keywords"]:
        keywords.update(keywords_from(str(keyword)))
    if not keywords:
        keywords = keywords_from(expected_answer)
    if not keywords:
        return maximum if normalize(student_answer) == normalize(expected_answer) else 0.0

    matched = keywords.intersection(keywords_from(student_answer))
    return maximum * len(matched) / len(keywords)


def evaluate(answer_key_text: str, student_text: str, pass_percent: float = 40) -> dict:
    answer_key = parse_answer_key(answer_key_text)
    if not answer_key:
        raise ValueError(
            "No answer key questions found. Use blocks such as "
            "'Q1 [2 marks] [type: mcq]' followed by 'Answer: B'."
        )

    student_answers = split_questions(student_text)
    results = []
    total = 0.0
    maximum = 0.0

    for number, question in answer_key.items():
        student_answer = student_answers.get(number, "")
        question_maximum = float(question["marks"])
        awarded = grade_answer(question, student_answer)
        total += awarded
        maximum += question_maximum
        results.append({
            "question": number,
            "type": question["type"],
            "awarded": round(awarded, 2),
            "maximum": question_maximum,
            "student_answer": student_answer,
        })

    percent = total / maximum * 100 if maximum else 0.0
    return {
        "total": round(total, 2),
        "maximum": round(maximum, 2),
        "percent": round(percent, 2),
        "result": "PASS" if percent >= pass_percent else "FAIL",
        "questions": results,
    }


@app.get("/")
def index():
    return send_from_directory(APP_DIRECTORY, "Frontend.html")


@app.post("/api/evaluate")
def evaluate_uploads():
    answer_key_file = request.files.get("answer_key")
    answer_sheet_file = request.files.get("answer_sheet")
    if not answer_key_file or not answer_sheet_file:
        return jsonify(error="Please upload both PDF files."), 400
    if (
        not answer_key_file.filename.lower().endswith(".pdf")
        or not answer_sheet_file.filename.lower().endswith(".pdf")
    ):
        return jsonify(error="Both uploaded files must be PDFs."), 400

    try:
        pass_percent = float(request.form.get("pass_percent", "40"))
    except ValueError:
        return jsonify(error="Pass mark must be a number between 0 and 100."), 400
    if not 0 <= pass_percent <= 100:
        return jsonify(error="Pass mark must be between 0 and 100."), 400

    try:
        answer_key_text = extract_pdf_text(
            answer_key_file.stream, answer_key_file.filename
        )
        student_text = extract_pdf_text(
            answer_sheet_file.stream, answer_sheet_file.filename
        )
        result = evaluate(answer_key_text, student_text, pass_percent)
    except (OSError, PdfReadError, ValueError) as error:
        return jsonify(error=str(error)), 400

    student_name = request.form.get("student", "").strip()
    if not student_name:
        name_match = re.search(
            r"(?im)^\s*(?:student|name)\s*:\s*(.+)$", student_text
        )
        student_name = name_match.group(1).strip() if name_match else "Student"

    formatted_results = []
    for question in result["questions"]:
        answer = question["student_answer"]
        question_type = question["type"]
        if question_type in OBJECTIVE_TYPES:
            feedback = (
                "Correct."
                if question["awarded"] == question["maximum"]
                else "Incorrect."
            )
        elif not answer:
            feedback = "No answer detected. Written-answer marks use keyword matching."
        else:
            feedback = "Written-answer marks use keyword matching."
        formatted_results.append({
            "qno": question["question"],
            "type": question_type,
            "awarded": question["awarded"],
            "maximum": question["maximum"],
            "feedback": feedback,
            "flagged": question_type not in OBJECTIVE_TYPES,
        })

    return jsonify(
        student=student_name,
        total=result["total"],
        maximum=result["maximum"],
        percent=result["percent"],
        result=result["result"],
        pass_percent=pass_percent,
        provisional=any(
            question["type"] not in OBJECTIVE_TYPES
            for question in result["questions"]
        ),
        results=formatted_results,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local answer-sheet website or grade two PDFs from the command line."
    )
    parser.add_argument("answer_key", nargs="?", help="Path to the answer-key PDF")
    parser.add_argument("answer_sheet", nargs="?", help="Path to the student's PDF")
    parser.add_argument(
        "--pass-percent",
        type=float,
        default=40,
        help="Percentage needed to pass (default: 40)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Local website port (default: 5000)",
    )
    args = parser.parse_args()

    if not args.answer_key and not args.answer_sheet:
        print("Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.")
        app.run(host="127.0.0.1", port=args.port)
        return 0
    if not args.answer_key or not args.answer_sheet:
        parser.error("provide both PDF paths, or provide neither to start the website")
    if not 0 <= args.pass_percent <= 100:
        parser.error("--pass-percent must be between 0 and 100")

    try:
        answer_key_text = extract_pdf_text(args.answer_key)
        student_text = extract_pdf_text(args.answer_sheet)
        result = evaluate(answer_key_text, student_text, args.pass_percent)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Score: {result['total']} / {result['maximum']}")
    print(f"Percentage: {result['percent']}%")
    print(f"Result: {result['result']}")
    print("\nQuestion-wise marks:")
    for question in result["questions"]:
        print(
            f"Q{question['question']}: {question['awarded']} / "
            f"{question['maximum']} ({question['type']})"
        )
    print(
        "\nNote: written answers use simple keyword matching, not "
        "human-level understanding."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
