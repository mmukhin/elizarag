"""Write data/companies.json: ticker, name and CIK from the filing headers, plus hand-curated aliases.

`aliases` match case-insensitively. `ambiguous_aliases` are ordinary words too ("apple pie", "target market")
and only match when capitalised as in the list. Edit ALIASES here and re-run; the JSON is generated.

Usage: python3 scripts/build_companies.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no PYTHONPATH

from app.edgar.chunking import CORPUS
from app.edgar.loader import parse_header
from app.edgar.query_filters import COMPANIES

# ticker -> (aliases, ambiguous_aliases)
ALIASES = {
    "AAPL": (["Apple Inc"], ["Apple"]),
    "ABBV": (["AbbVie"], []),
    "ADBE": (["Adobe"], []),
    "AMD": (["Advanced Micro Devices"], []),
    "AMZN": (["Amazon", "Amazon.com", "AWS"], []),
    "AXP": (["American Express", "Amex"], []),
    "BA": (["Boeing"], []),
    "BAC": (["Bank of America", "BofA"], []),
    "BLK": (["BlackRock"], []),
    "BRK": (["Berkshire Hathaway", "Berkshire"], []),
    "CAT": ([], ["Caterpillar"]),
    "CMCSA": (["Comcast", "NBCUniversal"], []),
    "COST": (["Costco"], []),
    "CRM": ([], ["Salesforce"]),
    "CSCO": (["Cisco"], []),
    "CVX": ([], ["Chevron"]),
    "DE": (["Deere", "John Deere"], []),
    "DIS": (["Disney", "Walt Disney"], []),
    "GE": (["General Electric", "GE Aerospace"], []),
    "GOOG": (["Google", "YouTube"], ["Alphabet"]),
    "GS": (["Goldman Sachs", "Goldman"], []),
    "HD": (["Home Depot"], []),
    "IBM": (["International Business Machines"], []),
    "INTC": (["Intel"], []),
    "JNJ": (["Johnson & Johnson", "Johnson and Johnson", "J&J"], []),
    "JPM": (["JPMorgan", "JP Morgan", "J.P. Morgan", "JPMorgan Chase"], ["Chase"]),
    "KO": (["Coca-Cola", "Coca Cola"], ["Coke"]),
    "LLY": (["Eli Lilly"], ["Lilly"]),
    "LMT": (["Lockheed Martin", "Lockheed"], []),
    "MA": (["Mastercard"], []),
    "MCD": (["McDonald's", "McDonalds", "McDonald’s"], []),
    "META": (["Meta Platforms", "Facebook", "Instagram"], ["Meta"]),
    "MRK": (["Merck"], []),
    "MS": (["Morgan Stanley"], []),
    "MSFT": (["Microsoft"], []),
    "NFLX": (["Netflix"], []),
    "NKE": (["Nike"], []),
    "NVDA": (["NVIDIA"], []),
    "ORCL": ([], ["Oracle"]),
    "PEP": (["PepsiCo", "Pepsi"], []),
    "PFE": (["Pfizer"], []),
    "PG": (["Procter & Gamble", "Procter and Gamble", "P&G"], []),
    "RTX": (["Raytheon"], []),
    "SBUX": (["Starbucks"], []),
    "T": (["AT&T"], []),
    "TGT": ([], ["Target"]),
    "TMO": (["Thermo Fisher"], []),
    "TSLA": (["Tesla"], []),
    "UNH": (["UnitedHealth", "United Health", "UnitedHealthcare"], []),
    "UPS": (["United Parcel Service"], []),
    "V": ([], ["Visa"]),
    "VZ": (["Verizon"], []),
    "WMT": (["Walmart", "Wal-Mart"], []),
    "XOM": (["Exxon Mobil", "ExxonMobil", "Exxon"], []),
}


def main() -> None:
    companies = {}
    for path in sorted(CORPUS.glob("*_full.txt")):
        fields, _ = parse_header(path.read_text(encoding="utf-8")[:4000])
        ticker = fields.get("ticker", "")
        if ticker and ticker not in companies:
            aliases, ambiguous = ALIASES.get(ticker, ([], []))
            companies[ticker] = {
                "company": fields.get("company", ""),
                "cik": fields.get("cik", ""),
                "aliases": aliases,
                "ambiguous_aliases": ambiguous,
            }
    if missing := sorted(set(companies) - set(ALIASES)):
        print(f"warning: no aliases for {missing}", file=sys.stderr)
    COMPANIES.write_text(json.dumps(dict(sorted(companies.items())), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(companies)} companies to {COMPANIES}")


if __name__ == "__main__":
    main()
