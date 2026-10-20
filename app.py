"""Веб-сервис UX-рецензента (FastAPI): страница с формой + API над agent.run().

POST /api/review {"url": "..."} → UX-отчёт (JSON)
GET  /api/limits                 → сколько проверок осталось сегодня
GET  /health                     → «живой ли сервис» (для Docker)

Каждая проверка стоит денег (gpt-4o), а сервис открыт всем, поэтому:
- лимит проверок в сутки с одного адреса и общий дневной потолок (сутки — по Москве);
- одна и та же страница за последние 24 часа отдаётся из кэша, лимит не тратит;
- не больше 2 проверок одновременно.
IP посетителей не хранится — только хэш с солью.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import agent

DATA = Path(os.environ.get("UX_DATA_DIR") or Path(__file__).parent / "data")
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / "ux.db"
PER_IP = int(os.environ.get("UX_LIMIT_PER_IP", "10"))
PER_DAY = int(os.environ.get("UX_LIMIT_PER_DAY", "60"))
CACHE_HOURS = 24
TRUST_PROXY = os.environ.get("UX_TRUST_PROXY") == "1"  # за обратным прокси (домен + HTTPS)
SALT = os.environ.get("UX_IP_SALT", "ux-reviewer")
MSK = timezone(timedelta(hours=3))
STATIC = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ux.app")
app = FastAPI(title="UX-рецензент сайта", version="1.0")
_slots = threading.BoundedSemaphore(2)


# ---------- база: журнал проверок + кэш ----------

def _db() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


with _db() as _c:
    _c.execute("""CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, day TEXT NOT NULL, ip TEXT NOT NULL,
        url TEXT NOT NULL, ok INTEGER NOT NULL, cached INTEGER NOT NULL DEFAULT 0,
        seconds REAL, report TEXT, error TEXT)""")
    _c.execute("CREATE INDEX IF NOT EXISTS reviews_day ON reviews(day, ip)")


def _today() -> str:
    return datetime.now(MSK).strftime("%Y-%m-%d")


def _ip(request: Request) -> str:
    ip = request.client.host if request.client else "?"
    if TRUST_PROXY and request.headers.get("x-forwarded-for"):
        ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    return hashlib.sha256((SALT + ip).encode()).hexdigest()[:16]


def _used(ip: str) -> tuple[int, int]:
    """Сколько платных (не из кэша) проверок сегодня: (с этого адреса, всего)."""
    with _db() as c:
        mine = c.execute("SELECT COUNT(*) FROM reviews WHERE day=? AND ip=? AND cached=0 AND ok=1",
                         (_today(), ip)).fetchone()[0]
        total = c.execute("SELECT COUNT(*) FROM reviews WHERE day=? AND cached=0 AND ok=1",
                          (_today(),)).fetchone()[0]
    return int(mine), int(total)


def _norm(url: str) -> str:
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    p = urlsplit(url)
    try:
        port = p.port
    except ValueError:  # «порт» не число — пусть агент сам сообщит понятную ошибку
        return url
    host = (p.hostname or "").lower() + (f":{port}" if port else "")
    return urlunsplit((p.scheme.lower(), host, p.path.rstrip("/") or "/", p.query, ""))


def _cached(url: str) -> dict[str, Any] | None:
    since = (datetime.now(MSK) - timedelta(hours=CACHE_HOURS)).isoformat()
    with _db() as c:
        row = c.execute("SELECT report FROM reviews WHERE url=? AND ok=1 AND cached=0 AND ts>? "
                        "ORDER BY id DESC LIMIT 1", (url, since)).fetchone()
    return json.loads(row["report"]) if row else None


def _save(ip: str, url: str, ok: bool, cached: bool = False, report: dict[str, Any] | None = None,
          error: str | None = None) -> None:
    with _db() as c:
        c.execute("INSERT INTO reviews(ts, day, ip, url, ok, cached, seconds, report, error) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (datetime.now(MSK).isoformat(), _today(), ip, url, int(ok), int(cached),
                   (report or {}).get("seconds"), json.dumps(report, ensure_ascii=False) if report else None,
                   error))


# ---------- API ----------

class ReviewIn(BaseModel):
    url: str = Field(min_length=4, max_length=500, description="адрес страницы")


def _limits(ip: str) -> dict[str, int]:
    mine, total = _used(ip)
    return {"left": max(0, min(PER_IP - mine, PER_DAY - total)), "per_ip": PER_IP}


@app.get("/api/limits")
def limits(request: Request) -> dict[str, int]:
    return _limits(_ip(request))


@app.post("/api/review")
def review(body: ReviewIn, request: Request) -> JSONResponse:
    ip, url = _ip(request), _norm(body.url)
    hit = _cached(url)
    if hit:
        _save(ip, url, ok=True, cached=True, report=hit)
        log.info("review cache hit %s", url)
        return JSONResponse({**hit, "cached": True, "limits": _limits(ip)})
    mine, total = _used(ip)
    if mine >= PER_IP:
        raise HTTPException(429, f"Лимит — {PER_IP} проверок в сутки с одного адреса. Приходите завтра.")
    if total >= PER_DAY:
        raise HTTPException(429, "Дневной лимит сервиса исчерпан. Приходите завтра.")
    if not _slots.acquire(timeout=60):
        raise HTTPException(503, "Сейчас идут другие проверки — попробуйте через минуту.")
    try:
        rep = agent.run(url).to_dict()
    except agent.PageError as e:
        _save(ip, url, ok=False, error=str(e))
        log.info("review page error %s: %s", url, e)
        raise HTTPException(422, str(e)) from None
    except Exception as e:  # noqa: BLE001 — нейросеть/сеть: пользователю коротко, в журнал подробно
        _save(ip, url, ok=False, error=f"{type(e).__name__}: {e}"[:500])
        log.exception("review failed %s", url)
        raise HTTPException(502, "Нейросеть не ответила — попробуйте ещё раз чуть позже.") from None
    finally:
        _slots.release()
    _save(ip, url, ok=True, report=rep)
    log.info("review ok %s: score %s, %ss, tokens %s/%s", url, rep["score"], rep["seconds"],
             rep["tokens_in"], rep["tokens_out"])
    return JSONResponse({**rep, "cached": False, "limits": _limits(ip)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
