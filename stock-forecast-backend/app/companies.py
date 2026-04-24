"""Static catalog of Russian stock market companies (tickers + sectors)."""
from __future__ import annotations

from typing import Dict, List, TypedDict

SECTORS: Dict[str, Dict[str, str]] = {
    "oil":       {"label": "Нефтегаз",      "color": "#f97316"},
    "finance":   {"label": "Финансы",       "color": "#3b82f6"},
    "tech":      {"label": "Технологии",    "color": "#a855f7"},
    "metal":     {"label": "Металлургия",   "color": "#94a3b8"},
    "retail":    {"label": "Ритейл",        "color": "#ec4899"},
    "telecom":   {"label": "Телеком",       "color": "#06b6d4"},
    "energy":    {"label": "Энергетика",    "color": "#eab308"},
    "transport": {"label": "Транспорт",     "color": "#22c55e"},
    "realty":    {"label": "Недвижимость",  "color": "#14b8a6"},
    "chem":      {"label": "Химия",         "color": "#84cc16"},
    "holding":   {"label": "Холдинги",      "color": "#6366f1"},
    "timber":    {"label": "Лесопром",      "color": "#10b981"},
    "agro":      {"label": "Агро",          "color": "#16a34a"},
    "machine":   {"label": "Машиностр.",    "color": "#64748b"},
}


class Company(TypedDict):
    ticker: str
    name: str
    sector: str


COMPANIES: List[Company] = [
    {"ticker": "SBER",  "name": "Сбербанк",              "sector": "finance"},
    {"ticker": "GAZP",  "name": "Газпром",                "sector": "oil"},
    {"ticker": "LKOH",  "name": "Лукойл",                 "sector": "oil"},
    {"ticker": "ROSN",  "name": "Роснефть",               "sector": "oil"},
    {"ticker": "GMKN",  "name": "Норильский никель",      "sector": "metal"},
    {"ticker": "NVTK",  "name": "Новатэк",                "sector": "oil"},
    {"ticker": "YNDX",  "name": "Яндекс",                 "sector": "tech"},
    {"ticker": "MTSS",  "name": "МТС",                    "sector": "telecom"},
    {"ticker": "MGNT",  "name": "Магнит",                 "sector": "retail"},
    {"ticker": "FIVE",  "name": "X5 Retail Group",        "sector": "retail"},
    {"ticker": "PLZL",  "name": "Полюс Золото",           "sector": "metal"},
    {"ticker": "POLY",  "name": "Polymetal",              "sector": "metal"},
    {"ticker": "ALRS",  "name": "АЛРОСА",                 "sector": "metal"},
    {"ticker": "CHMF",  "name": "Северсталь",             "sector": "metal"},
    {"ticker": "NLMK",  "name": "НЛМК",                   "sector": "metal"},
    {"ticker": "MAGN",  "name": "ММК",                    "sector": "metal"},
    {"ticker": "VTBR",  "name": "ВТБ",                    "sector": "finance"},
    {"ticker": "MOEX",  "name": "Московская биржа",       "sector": "finance"},
    {"ticker": "PHOR",  "name": "ФосАгро",                "sector": "chem"},
    {"ticker": "RUAL",  "name": "РУСАЛ",                  "sector": "metal"},
    {"ticker": "OZON",  "name": "Ozon",                   "sector": "tech"},
    {"ticker": "VKCO",  "name": "ВКонтакте",              "sector": "tech"},
    {"ticker": "IRAO",  "name": "Интер РАО",              "sector": "energy"},
    {"ticker": "FEES",  "name": "ФСК ЕЭС",                "sector": "energy"},
    {"ticker": "HYDR",  "name": "РусГидро",               "sector": "energy"},
    {"ticker": "RTKM",  "name": "Ростелеком",             "sector": "telecom"},
    {"ticker": "AFLT",  "name": "Аэрофлот",               "sector": "transport"},
    {"ticker": "PIKK",  "name": "ПИК",                    "sector": "realty"},
    {"ticker": "SMLT",  "name": "Самолёт",                "sector": "realty"},
    {"ticker": "SGZH",  "name": "Сегежа",                 "sector": "timber"},
    {"ticker": "MTLR",  "name": "Мечел",                  "sector": "metal"},
    {"ticker": "AFKS",  "name": "АФК Система",            "sector": "holding"},
    {"ticker": "CBOM",  "name": "МКБ",                    "sector": "finance"},
    {"ticker": "TATN",  "name": "Татнефть",               "sector": "oil"},
    {"ticker": "SNGS",  "name": "Сургутнефтегаз",         "sector": "oil"},
    {"ticker": "BANE",  "name": "Башнефть",               "sector": "oil"},
    {"ticker": "TRNFP", "name": "Транснефть",             "sector": "oil"},
    {"ticker": "NMTP",  "name": "НМТП",                   "sector": "transport"},
    {"ticker": "FLOT",  "name": "Совкомфлот",             "sector": "transport"},
    {"ticker": "UWGN",  "name": "Уралвагонзавод",         "sector": "machine"},
    {"ticker": "TCSG",  "name": "Т-Банк",                 "sector": "finance"},
    {"ticker": "FIXP",  "name": "Fix Price",              "sector": "retail"},
    {"ticker": "ENPG",  "name": "ЭН+",                    "sector": "energy"},
    {"ticker": "DSKY",  "name": "Детский мир",            "sector": "retail"},
    {"ticker": "LENT",  "name": "Лента",                  "sector": "retail"},
    {"ticker": "RASP",  "name": "Распадская",             "sector": "metal"},
    {"ticker": "SELG",  "name": "Селигдар",               "sector": "metal"},
    {"ticker": "POGR",  "name": "ПОЭЗ",                   "sector": "energy"},
    {"ticker": "BSPB",  "name": "Банк Санкт-Петербург",   "sector": "finance"},
    {"ticker": "AQUA",  "name": "Инарктика",              "sector": "agro"},
    {"ticker": "RNFT",  "name": "РуссНефть",              "sector": "oil"},
    {"ticker": "MSNG",  "name": "Мосэнерго",              "sector": "energy"},
    {"ticker": "LSRG",  "name": "ЛСР",                    "sector": "realty"},
    {"ticker": "RENI",  "name": "Ренессанс Страхование",  "sector": "finance"},
]


COMPANY_BY_TICKER: Dict[str, Company] = {c["ticker"]: c for c in COMPANIES}
