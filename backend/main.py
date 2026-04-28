from __future__ import annotations

import asyncio
import json
import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
STATE_FILE = DATA_DIR / "state.json"
TOKEN_FILE = DATA_DIR / "token.json"
load_dotenv(ROOT / ".env")

KIS_BASE_URL = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443").rstrip("/")
APP_KEY = os.getenv("KIS_APP_KEY", "")
APP_SECRET = os.getenv("KIS_APP_SECRET", "")
ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")
ACCOUNT_PRODUCT_CODE = os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "")
LIVE_TRADING = os.getenv("KIS_LIVE_TRADING", "false").lower() == "true"
DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "NAS")
AUTO_LOOP_INTERVAL_SECONDS = int(os.getenv("AUTO_LOOP_INTERVAL_SECONDS", "300"))

ALLOWED_SYMBOLS = {"SOXL", "TQQQ"}

app = FastAPI(title="KIS Infinite Buy Live Terminal")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class StrategyConfig(BaseModel):
    symbol: Literal["SOXL", "TQQQ"] = "SOXL"
    exchange: str = DEFAULT_EXCHANGE
    enabled: bool = False
    total_seed_usd: float = Field(default=350.0, gt=0)
    buy_amount_usd: float = Field(default=25.0, gt=0)
    max_buy_count: int = Field(default=14, ge=1, le=365)
    target_profit_pct: float = Field(default=10.0, gt=0, le=200)
    stop_loss_pct: float = Field(default=0.0, ge=0, le=100)  # 0이면 손절 비활성화
    market_order: bool = True
    fractional_order: bool = True  # true면 0.xx주 소수점 주문 시도


class State(BaseModel):
    config: StrategyConfig = StrategyConfig()
    position_qty: float = 0.0
    avg_price: float = 0.0
    invested_usd: float = 0.0
    buy_count: int = 0
    last_buy_date: str | None = None
    cycle_no: int = 1
    logs: list[dict[str, Any]] = []


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> State:
    if not STATE_FILE.exists():
        DATA_DIR.mkdir(exist_ok=True)
        st = State()
        save_state(st)
        return st
    return State(**json.loads(STATE_FILE.read_text(encoding="utf-8")))


def save_state(st: State) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(st.model_dump_json(indent=2), encoding="utf-8")


def add_log(st: State, level: str, message: str, extra: dict[str, Any] | None = None) -> None:
    st.logs.insert(0, {"time": now_iso(), "level": level, "message": message, "extra": extra or {}})
    st.logs = st.logs[:200]


def require_env() -> None:
    missing = []
    for key, value in {
        "KIS_APP_KEY": APP_KEY,
        "KIS_APP_SECRET": APP_SECRET,
        "KIS_ACCOUNT_NO": ACCOUNT_NO,
        "KIS_ACCOUNT_PRODUCT_CODE": ACCOUNT_PRODUCT_CODE,
    }.items():
        if not value:
            missing.append(key)
    if missing:
        raise HTTPException(status_code=400, detail=f".env 누락: {', '.join(missing)}")


async def get_token(force_refresh: bool = False) -> str:
    require_env()
    DATA_DIR.mkdir(exist_ok=True)

    if TOKEN_FILE.exists() and not force_refresh:
        try:
            cached = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
            # 만료 5분 전부터 자동 재발급
            if cached.get("access_token") and cached.get("expires_at", 0) > time.time() + 300:
                return cached["access_token"]
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{KIS_BASE_URL}/oauth2/tokenP",
            json={"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET},
            headers={"content-type": "application/json"},
        )
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"KIS 토큰 발급 실패: {r.text}")
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise HTTPException(status_code=502, detail=f"KIS 토큰 응답 이상: {data}")
    expires_in = int(data.get("expires_in", 86400))
    TOKEN_FILE.write_text(
        json.dumps({"access_token": token, "expires_at": time.time() + expires_in - 300}),
        encoding="utf-8",
    )
    return token

async def kis_headers(tr_id: str, with_hash: dict[str, Any] | None = None) -> dict[str, str]:
    token = await get_token()
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
    }
    if with_hash is not None:
        headers["hashkey"] = await hashkey(with_hash)
    return headers


async def hashkey(body: dict[str, Any]) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{KIS_BASE_URL}/uapi/hashkey",
            json=body,
            headers={"content-type": "application/json", "appkey": APP_KEY, "appsecret": APP_SECRET},
        )
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"KIS hashkey 실패: {r.text}")
    return r.json().get("HASH", "")


async def overseas_price(symbol: str, exchange: str) -> dict[str, Any]:
    symbol = symbol.upper()
    if symbol not in ALLOWED_SYMBOLS:
        raise HTTPException(status_code=400, detail="SOXL/TQQQ만 허용")

    params = {"AUTH": "", "EXCD": exchange, "SYMB": symbol}
    async with httpx.AsyncClient(timeout=15) as client:
        headers = await kis_headers("HHDFS00000300")
        r = await client.get(f"{KIS_BASE_URL}/uapi/overseas-price/v1/quotations/price", headers=headers, params=params)

        # 토큰 만료/무효 응답이면 토큰 강제 재발급 후 1회 재시도
        if r.status_code in (401, 403):
            try:
                TOKEN_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            await get_token(force_refresh=True)
            headers = await kis_headers("HHDFS00000300")
            r = await client.get(f"{KIS_BASE_URL}/uapi/overseas-price/v1/quotations/price", headers=headers, params=params)

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"KIS 해외 현재가 실패: {r.text}")
    data = r.json()
    out = data.get("output") or {}
    last = out.get("last") or out.get("ovrs_nmix_prpr") or out.get("stck_prpr")
    try:
        price = float(last)
    except Exception:
        raise HTTPException(status_code=502, detail=f"현재가 파싱 실패: {data}")
    return {"symbol": symbol, "exchange": exchange, "price": price, "raw": out, "fetched_at": now_iso()}

def format_qty(qty: float, fractional: bool = True) -> str:
    if fractional:
        text = f"{qty:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(int(math.floor(qty)))


async def place_overseas_order(side: Literal["buy", "sell"], symbol: str, exchange: str, qty: float, price: float, market_order: bool, fractional_order: bool = True) -> dict[str, Any]:
    if qty <= 0:
        raise HTTPException(status_code=400, detail="주문 수량이 0 이하")
    if not LIVE_TRADING:
        raise HTTPException(status_code=403, detail="실전 주문 차단됨: .env에 KIS_LIVE_TRADING=true를 넣어야 실제 주문이 나갑니다.")

    # KIS 해외주식 주문. 미국주식 시장가/지정가 구분은 계좌/시장/시간대에 따라 제한될 수 있음.
    tr_id = "JTTT1002U" if side == "buy" else "JTTT1006U"
    ord_dvsn = "01" if market_order else "00"  # 01 시장가, 00 지정가로 쓰이는 케이스가 많음. 실패 시 KIS 응답 확인.
    order_price = "0" if market_order else f"{price:.2f}"
    body = {
        "CANO": ACCOUNT_NO,
        "ACNT_PRDT_CD": ACCOUNT_PRODUCT_CODE,
        "OVRS_EXCG_CD": exchange,
        "PDNO": symbol,
        "ORD_QTY": format_qty(qty, fractional_order),
        "OVRS_ORD_UNPR": order_price,
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": ord_dvsn,
    }
    headers = await kis_headers(tr_id, with_hash=body)
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{KIS_BASE_URL}/uapi/overseas-stock/v1/trading/order", headers=headers, json=body)

        # 토큰 만료/무효 응답이면 토큰 강제 재발급 후 1회 재시도
        if r.status_code in (401, 403):
            try:
                TOKEN_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            await get_token(force_refresh=True)
            headers = await kis_headers(tr_id, with_hash=body)
            r = await client.post(f"{KIS_BASE_URL}/uapi/overseas-stock/v1/trading/order", headers=headers, json=body)

    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}
    if r.status_code >= 400 or data.get("rt_cd") not in (None, "0"):
        hint = ""
        if fractional_order and "." in body.get("ORD_QTY", ""):
            hint = " / 소수점 주문이 계좌 또는 KIS 해외주식 주문 API에서 거부됐을 수 있습니다. 한투 앱에서 해외주식 소수점 거래 신청 여부를 확인하거나 소수점 주문을 끄고 1주 이상 주문으로 사용하세요."
        raise HTTPException(status_code=502, detail=f"KIS 주문 실패{hint}: {data}")
    return {"request": body, "response": data}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/state")
def api_state():
    st = load_state()
    return {**st.model_dump(), "live_trading": LIVE_TRADING, "env_ready": bool(APP_KEY and APP_SECRET and ACCOUNT_NO and ACCOUNT_PRODUCT_CODE)}


@app.get("/api/token/status")
def api_token_status():
    if not TOKEN_FILE.exists():
        return {"cached": False, "message": "아직 발급된 토큰 캐시 없음"}
    try:
        cached = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        expires_at = float(cached.get("expires_at", 0))
        return {
            "cached": True,
            "expires_at": expires_at,
            "seconds_left": max(0, int(expires_at - time.time())),
        }
    except Exception as e:
        return {"cached": False, "error": str(e)}


@app.post("/api/token/refresh")
async def api_token_refresh():
    token = await get_token(force_refresh=True)
    return {"ok": True, "message": "KIS access token 자동 갱신 완료", "token_tail": token[-6:]}


@app.post("/api/config")
def api_config(cfg: StrategyConfig):
    st = load_state()
    if cfg.symbol not in ALLOWED_SYMBOLS:
        raise HTTPException(status_code=400, detail="SOXL/TQQQ만 가능")
    st.config = cfg
    add_log(st, "info", "설정 저장", cfg.model_dump())
    save_state(st)
    return st


@app.post("/api/reset-cycle")
def api_reset_cycle():
    st = load_state()
    st.position_qty = 0.0
    st.avg_price = 0.0
    st.invested_usd = 0.0
    st.buy_count = 0
    st.last_buy_date = None
    st.cycle_no += 1
    add_log(st, "warn", "사이클 수동 초기화")
    save_state(st)
    return st


@app.get("/api/quote/{symbol}")
async def api_quote(symbol: str):
    st = load_state()
    return await overseas_price(symbol.upper(), st.config.exchange)


async def evaluate_strategy(trigger: str = "manual") -> dict[str, Any]:
    st = load_state()
    cfg = st.config
    quote = await overseas_price(cfg.symbol, cfg.exchange)
    current = float(quote["price"])

    if not cfg.enabled:
        add_log(st, "info", f"자동매매 OFF: 평가 중단 ({trigger})", {"price": current})
        save_state(st)
        return {"action": "none", "reason": "disabled", "quote": quote, "state": st.model_dump()}

    if st.position_qty > 0 and st.avg_price > 0:
        pnl_pct = (current / st.avg_price - 1) * 100
        target_price = st.avg_price * (1 + cfg.target_profit_pct / 100)
        if pnl_pct >= cfg.target_profit_pct:
            result = await place_overseas_order("sell", cfg.symbol, cfg.exchange, st.position_qty, current, cfg.market_order, cfg.fractional_order)
            sold_qty = st.position_qty
            st.position_qty = 0.0
            st.avg_price = 0.0
            st.invested_usd = 0.0
            st.buy_count = 0
            st.last_buy_date = None
            st.cycle_no += 1
            add_log(st, "success", f"목표수익 도달: {format_qty(sold_qty, cfg.fractional_order)}주 전량 매도 주문 ({trigger})", {"pnl_pct": pnl_pct, "target_price": target_price, "order": result})
            save_state(st)
            return {"action": "sell", "quote": quote, "order": result, "state": st.model_dump()}

        if cfg.stop_loss_pct > 0 and pnl_pct <= -cfg.stop_loss_pct:
            result = await place_overseas_order("sell", cfg.symbol, cfg.exchange, st.position_qty, current, cfg.market_order, cfg.fractional_order)
            sold_qty = st.position_qty
            st.position_qty = 0.0
            st.avg_price = 0.0
            st.invested_usd = 0.0
            st.buy_count = 0
            st.last_buy_date = None
            st.config.enabled = False
            st.cycle_no += 1
            add_log(st, "error", f"손절 조건 도달: -{cfg.stop_loss_pct}% / {format_qty(sold_qty, cfg.fractional_order)}주 전량 매도 후 자동매매 중단 ({trigger})", {"pnl_pct": pnl_pct, "stop_loss_pct": cfg.stop_loss_pct, "order": result})
            save_state(st)
            return {"action": "stop_loss_sell", "quote": quote, "order": result, "state": st.model_dump()}

    today = date.today().isoformat()
    if st.last_buy_date == today:
        add_log(st, "info", f"오늘 이미 매수함 ({trigger})", {"price": current})
        save_state(st)
        return {"action": "none", "reason": "already_bought_today", "quote": quote, "state": st.model_dump()}

    if st.buy_count >= cfg.max_buy_count:
        add_log(st, "warn", f"최대 매수 횟수 도달 ({trigger})", {"buy_count": st.buy_count})
        save_state(st)
        return {"action": "none", "reason": "max_buy_count", "quote": quote, "state": st.model_dump()}

    remaining_seed = max(cfg.total_seed_usd - st.invested_usd, 0)
    amount = min(cfg.buy_amount_usd, remaining_seed)
    raw_qty = amount / current
    qty = round(raw_qty, 6) if cfg.fractional_order else math.floor(raw_qty)
    if qty <= 0:
        add_log(st, "warn", "매수 가능 수량 0주: 매수금/현재가 계산 결과 0", {"amount": amount, "price": current, "fractional_order": cfg.fractional_order})
        save_state(st)
        return {"action": "none", "reason": "qty_zero", "quote": quote, "state": st.model_dump()}

    result = await place_overseas_order("buy", cfg.symbol, cfg.exchange, qty, current, cfg.market_order, cfg.fractional_order)
    new_cost = qty * current
    total_cost = st.invested_usd + new_cost
    total_qty = st.position_qty + qty
    st.avg_price = total_cost / total_qty
    st.invested_usd = total_cost
    st.position_qty = total_qty
    st.buy_count += 1
    st.last_buy_date = today
    add_log(st, "success", f"정액 매수 주문: {cfg.symbol} {format_qty(qty, cfg.fractional_order)}주 ({trigger})", {"price": current, "amount_usd": amount, "order": result})
    save_state(st)
    return {"action": "buy", "quote": quote, "order": result, "state": st.model_dump()}


@app.post("/api/evaluate")
async def api_evaluate():
    return await evaluate_strategy("manual")


@app.post("/api/bot/start")
def api_bot_start():
    st = load_state()
    st.config.enabled = True
    add_log(st, "success", "자동매매 시작: 서버 백그라운드 루프 ON")
    save_state(st)
    return st


@app.post("/api/bot/stop")
def api_bot_stop():
    st = load_state()
    st.config.enabled = False
    add_log(st, "warn", "자동매매 중단: 서버 백그라운드 루프 OFF")
    save_state(st)
    return st


async def auto_loop():
    while True:
        try:
            st = load_state()
            if st.config.enabled:
                await evaluate_strategy("auto-loop")
        except Exception as e:
            try:
                st = load_state()
                add_log(st, "error", f"자동 루프 오류: {e}")
                save_state(st)
            except Exception:
                pass
        await asyncio.sleep(AUTO_LOOP_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_loop())


@app.post("/api/manual-buy")
async def api_manual_buy():
    st = load_state()
    cfg = st.config
    quote = await overseas_price(cfg.symbol, cfg.exchange)
    current = float(quote["price"])
    raw_qty = cfg.buy_amount_usd / current
    qty = round(raw_qty, 6) if cfg.fractional_order else math.floor(raw_qty)
    result = await place_overseas_order("buy", cfg.symbol, cfg.exchange, qty, current, cfg.market_order, cfg.fractional_order)
    add_log(st, "success", f"수동 매수 주문 전송: {format_qty(qty, cfg.fractional_order)}주", {"price": current, "amount_usd": cfg.buy_amount_usd, "order": result})
    save_state(st)
    return {"quote": quote, "order": result}


@app.post("/api/manual-sell-all")
async def api_manual_sell_all():
    st = load_state()
    cfg = st.config
    if st.position_qty <= 0:
        raise HTTPException(status_code=400, detail="앱 기록상 보유수량이 없습니다.")
    quote = await overseas_price(cfg.symbol, cfg.exchange)
    current = float(quote["price"])
    result = await place_overseas_order("sell", cfg.symbol, cfg.exchange, st.position_qty, current, cfg.market_order, cfg.fractional_order)
    add_log(st, "success", f"수동 전량 매도 주문 전송: {format_qty(st.position_qty, cfg.fractional_order)}주", {"price": current, "order": result})
    st.position_qty = 0.0
    st.avg_price = 0.0
    st.invested_usd = 0.0
    st.buy_count = 0
    st.last_buy_date = None
    st.cycle_no += 1
    save_state(st)
    return {"quote": quote, "order": result, "state": st.model_dump()}
