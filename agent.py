"""UX-рецензент сайта: ссылка → текст и структура страницы → gpt-4o → отчёт.

Отчёт: оценка 1–10, вывод, достоинства, слабые места и ровно 5 рекомендаций с приоритетом.
Запуск:  python agent.py https://example.com          — отчёт текстом
         python agent.py https://example.com --json   — отчёт в JSON
Из кода: run(url) -> UXReport
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Tag

from openai_module import BadAnswer, LLMError, Usage, ask_json

log = logging.getLogger("ux.agent")

MAX_BYTES = 3_000_000      # страницы тяжелее не качаем
MAX_TEXT = 7000            # столько знаков видимого текста уходит модели
MAX_REDIRECTS = 5
FETCH_TIMEOUT = 15
UA = "Mozilla/5.0 (compatible; UX-Reviewer/1.0; +https://github.com/JohnSmith-SG/ux-reviewer)"


class PageError(ValueError):
    """Страницу нельзя загрузить или разобрать — понятное сообщение для пользователя."""


# ---------- загрузка: только публичные адреса (защита от SSRF) ----------

def _check_public(url: str) -> str:
    """Пропускаем только http(s) на стандартных портах и хосты с публичными IP —
    иначе через сервис можно было бы заглянуть во внутреннюю сеть сервера."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise PageError("Нужна ссылка вида https://сайт.ru")
    if parts.port not in (None, 80, 443):
        raise PageError("Поддерживаются только стандартные порты 80 и 443")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise PageError(f"Сайт {parts.hostname} не найден — проверьте адрес") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise PageError("Этот адрес ведёт во внутреннюю сеть — такие сайты не проверяем")
    return url


def fetch(url: str) -> tuple[str, str]:
    """Скачивает HTML, сам проходит редиректы (каждый адрес проверяется). → (итоговый url, html)."""
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(url)
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "ru,en;q=0.8"},
                             timeout=FETCH_TIMEOUT, allow_redirects=False, stream=True)
        except requests.RequestException as e:
            raise PageError(f"Сайт не ответил: {type(e).__name__}") from None
        if r.is_redirect and r.headers.get("location"):
            url = urljoin(url, r.headers["location"])
            r.close()
            continue
        if r.status_code >= 400:
            raise PageError(f"Сайт ответил ошибкой {r.status_code}")
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype:
            raise PageError(f"По ссылке не веб-страница ({ctype or 'тип не указан'})")
        body = b""
        for chunk in r.iter_content(65536):
            body += chunk
            if len(body) > MAX_BYTES:
                raise PageError("Страница слишком большая (больше 3 МБ)")
        enc = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else None
        return url, body.decode(enc or r.apparent_encoding or "utf-8", errors="replace")
    raise PageError("Слишком много перенаправлений")


# ---------- что на странице важно для UX ----------

@dataclass
class PageFacts:
    url: str
    title: str = ""
    description: str = ""
    lang: str = ""
    has_viewport: bool = False
    headings: list[str] = field(default_factory=list)
    nav: list[str] = field(default_factory=list)
    ctas: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    images: int = 0
    images_no_alt: int = 0
    links: int = 0
    text: str = ""

    def as_prompt(self) -> str:
        def lst(items: list[str], limit: int) -> str:
            return "; ".join(items[:limit]) or "—"
        facts = [
            f"- адаптация под телефоны (meta viewport): {'ЕСТЬ' if self.has_viewport else 'НЕТ'}",
            f"- картинок {self.images}, без описания (alt): {self.images_no_alt}"
            + (" — у всех картинок alt ЕСТЬ" if self.images and not self.images_no_alt else ""),
            f"- форм на странице: {len(self.forms)}",
            f"- мета-описание: {'есть' if self.description else 'НЕТ'}; язык страницы: "
            f"{self.lang or 'НЕ указан'}",
            f"- заголовков H1: {sum(h.startswith('H1:') for h in self.headings)}",
        ]
        return "\n".join([
            "ПРОВЕРЕННЫЕ ФАКТЫ (посчитаны программой по HTML — не противоречь им):",
            *facts,
            "",
            f"Адрес: {self.url}",
            f"Заголовок страницы (title): {self.title or '— нет'}",
            f"Мета-описание: {self.description or '— нет'}",
            f"Язык (lang): {self.lang or '— не указан'}; адаптация под телефоны (meta viewport): "
            f"{'есть' if self.has_viewport else 'НЕТ'}",
            f"Заголовки по порядку: {lst(self.headings, 40)}",
            f"Меню / навигация: {lst(self.nav, 30)}",
            f"Кнопки и призывы к действию: {lst(self.ctas, 30)}",
            f"Формы: {lst(self.forms, 10)}",
            f"Картинок: {self.images}, из них без описания (alt): {self.images_no_alt}; ссылок: {self.links}",
            "",
            "Видимый текст страницы (сокращённо):",
            self.text,
        ])


def _clean(s: str, limit: int = 120) -> str:
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit - 1] + "…"


def extract(url: str, html: str) -> PageFacts:
    soup = BeautifulSoup(html, "html.parser")
    f = PageFacts(url=url)
    if soup.title and soup.title.string:
        f.title = _clean(soup.title.string, 200)
    meta = soup.find("meta", attrs={"name": "description"})
    if isinstance(meta, Tag):
        f.description = _clean(str(meta.get("content", "")), 300)
    root = soup.find("html")
    if isinstance(root, Tag):
        f.lang = str(root.get("lang", ""))
    f.has_viewport = soup.find("meta", attrs={"name": "viewport"}) is not None

    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    f.headings = [f"{h.name.upper()}: {_clean(h.get_text())}" for h in soup.find_all(["h1", "h2", "h3"])
                  if h.get_text(strip=True)]
    for nav in soup.find_all("nav"):
        f.nav += [_clean(a.get_text(), 40) for a in nav.find_all("a") if a.get_text(strip=True)]
    seen: set[str] = set()
    for el in soup.find_all(["button", "a", "input"]):
        if el.name == "input" and el.get("type") not in ("submit", "button"):
            continue
        cls = " ".join(el.get("class") or []).lower()
        if el.name == "a" and not any(k in cls for k in ("btn", "button", "cta")):
            continue
        t = _clean(el.get_text() or str(el.get("value", "")), 60)
        if t and t not in seen:
            seen.add(t)
            f.ctas.append(t)
    for form in soup.find_all("form"):
        fields = [str(i.get("placeholder") or i.get("name") or i.get("type") or "поле")
                  for i in form.find_all(["input", "textarea", "select"])
                  if i.get("type") not in ("hidden", "submit", "button")]
        f.forms.append(f"{len(fields)} полей ({', '.join(fields[:8])})")
    imgs = soup.find_all("img")
    f.images = len(imgs)
    f.images_no_alt = sum(1 for i in imgs if not str(i.get("alt", "")).strip())
    f.links = len(soup.find_all("a"))
    text = _clean(soup.get_text(" "), 10**9)
    f.text = text[:MAX_TEXT] + (" …[текст обрезан]" if len(text) > MAX_TEXT else "")
    if len(text) < 80 and not f.headings:
        raise PageError("На странице почти нет текста — возможно, сайт собирается скриптом "
                        "в браузере, а агент видит только исходный HTML")
    return f


# ---------- нейросеть ----------

SYSTEM = (
    "Ты — опытный UX-эксперт и редактор интерфейсов. Тебе дают текст и структуру одной "
    "веб-страницы (заголовки, меню, кнопки, формы, картинки). Визуального дизайна ты не видишь — "
    "оценивай то, что можно понять из текста и структуры: ясность ценности, иерархию информации, "
    "навигацию, призывы к действию, формы, доверие, доступность, адаптацию под телефоны.\n"
    "Правила:\n"
    "1. Каждое достоинство, слабое место и рекомендацию опирай на конкретику страницы — цитируй "
    "заголовки, тексты кнопок, пункты меню, цифры. Общие советы, подходящие любому сайту "
    "(«улучшить навигацию», «сократить текст»), без привязки к этой странице не годятся.\n"
    "2. Блок «ПРОВЕРЕННЫЕ ФАКТЫ» посчитан программой и точен. Не советуй то, что уже сделано: "
    "если alt есть у всех картинок — не предлагай добавить alt; если meta viewport есть — не пиши, "
    "что сайт не адаптирован под телефоны.\n"
    "3. Не выдумывай того, чего нет в данных; если данных мало — так и скажи.\n"
    "4. Ты не видишь цвета, размеры, шрифты и картинки — не пиши, что что-то «не выделено "
    "визуально» или «бедно выглядит». Прежде чем утверждать, что какого-то раздела или блока "
    "нет, проверь список заголовков: если он есть — оценивай, как он сделан, а не предлагай создать.\n"
    "5. Пиши на русском, просто и по делу — для владельца сайта, а не для дизайнера.\n"
    "Дай ровно 5 рекомендаций, от самой важной к менее важной; в каждой — что сделать, зачем, "
    "и в поле evidence — на каком факте или цитате со страницы она основана."
)

_ITEMS = {"type": "array", "items": {"type": "string"}}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["site_type", "score", "summary", "strengths", "weaknesses", "recommendations"],
    "properties": {
        "site_type": {"type": "string", "description": "что это за сайт, 2–5 слов по-русски"},
        "score": {"type": "integer", "description": "общая оценка UX от 1 до 10"},
        "summary": {"type": "string", "description": "вывод в 2–3 предложениях"},
        "strengths": {**_ITEMS, "description": "3–5 достоинств"},
        "weaknesses": {**_ITEMS, "description": "3–5 слабых мест"},
        "recommendations": {
            "type": "array",
            "description": "ровно 5 рекомендаций",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "details", "evidence", "priority"],
                "properties": {
                    "title": {"type": "string", "description": "что сделать, до 10 слов"},
                    "details": {"type": "string", "description": "как и зачем, 1–3 предложения"},
                    "evidence": {"type": "string",
                                 "description": "на чём основано: цитата или факт со страницы"},
                    "priority": {"type": "string", "enum": ["высокий", "средний", "низкий"]},
                },
            },
        },
    },
}


@dataclass
class Recommendation:
    title: str
    details: str
    evidence: str
    priority: str


@dataclass
class UXReport:
    url: str
    site_type: str
    score: int
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[Recommendation]
    model: str
    seconds: float
    tokens_in: int
    tokens_out: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(data: dict[str, Any]) -> None:
    recs = data.get("recommendations") or []
    if len(recs) != 5:
        raise BadAnswer(f"рекомендаций {len(recs)}, а нужно 5")
    if not data.get("strengths") or not data.get("weaknesses"):
        raise BadAnswer("нет достоинств или слабых мест")
    if not 1 <= int(data.get("score", 0)) <= 10:
        raise BadAnswer("оценка вне 1–10")


_CONTRADICTS = {
    # тема рекомендации → слова в заголовке, при которых она противоречит проверенному факту
    "alt": ("alt", "альтернативн", "описания к изображ", "описания к картин", "подписи к изображ"),
    "viewport": ("адаптировать под мобил", "мобильную версию", "адаптивн", "для мобильных устройств"),
}


def _check_facts(facts: PageFacts) -> Callable[[dict[str, Any]], None]:
    """Проверка ответа: формат + не советует ли модель то, что на странице уже есть."""
    def check(data: dict[str, Any]) -> None:
        _check(data)
        for r in data["recommendations"]:
            title = r["title"].lower()
            if facts.images and not facts.images_no_alt and any(w in title for w in _CONTRADICTS["alt"]):
                raise BadAnswer("совет добавить alt, хотя он есть у всех картинок")
            if facts.has_viewport and any(w in title for w in _CONTRADICTS["viewport"]) \
                    and "проверить" not in title:
                raise BadAnswer("совет про адаптацию под телефоны, хотя viewport есть")
    return check


def run(url: str) -> UXReport:
    """Главная функция агента: ссылка → UX-отчёт. PageError — проблема со страницей,
    LLMError и ошибки openai — проблема с нейросетью."""
    t0 = time.monotonic()
    final_url, html = fetch(url)
    facts = extract(final_url, html)
    log.info("page %s: %d заголовков, %d кнопок, %d знаков текста",
             final_url, len(facts.headings), len(facts.ctas), len(facts.text))
    data, usage = ask_json(SYSTEM, facts.as_prompt(), SCHEMA, name="ux_report",
                           check=_check_facts(facts))
    return _report(final_url, data, usage, time.monotonic() - t0)


def _report(url: str, d: dict[str, Any], u: Usage, secs: float) -> UXReport:
    return UXReport(
        url=url, site_type=d["site_type"], score=int(d["score"]), summary=d["summary"],
        strengths=list(d["strengths"]), weaknesses=list(d["weaknesses"]),
        recommendations=[Recommendation(**r) for r in d["recommendations"]],
        model=u.model, seconds=round(secs, 1), tokens_in=u.tokens_in, tokens_out=u.tokens_out)


# ---------- командная строка ----------

def _print(rep: UXReport) -> None:
    print(f"\nUX-отчёт: {rep.url}\n{rep.site_type} · оценка {rep.score}/10\n\n{rep.summary}\n")
    print("Достоинства:")
    for s in rep.strengths:
        print(f"  + {s}")
    print("\nСлабые места:")
    for s in rep.weaknesses:
        print(f"  − {s}")
    print("\n5 рекомендаций:")
    for i, r in enumerate(rep.recommendations, 1):
        print(f"  {i}. [{r.priority}] {r.title}\n     {r.details}\n     основание: {r.evidence}")
    print(f"\n{rep.model} · {rep.seconds} с · токены {rep.tokens_in}/{rep.tokens_out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="UX-рецензент сайта")
    p.add_argument("url", help="адрес страницы, например https://example.com")
    p.add_argument("--json", action="store_true", help="вывести отчёт в JSON")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        rep = run(a.url)
    except PageError as e:
        print(f"Страница: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — LLMError, ошибки openai: понятный выход без трассировки
        print(f"Нейросеть не ответила: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return 3
    if a.json:
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print(rep)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
