"""OSI option symbol construction for Webull market-data APIs."""

from __future__ import annotations

from datetime import date

from joker.schemas.options_data import OptionContractMetadata


def build_osi_symbol(
    underlying: str,
    expiration: date,
    strike: float,
    option_type: str,
) -> str:
    """Build Webull/OSI option symbol, e.g. SPY260701C00550000."""
    if underlying.upper() != "SPY":
        raise ValueError("Only SPY supported")
    cp = "C" if option_type.lower() in ("call", "c") else "P"
    strike_int = int(round(strike * 1000))
    return f"{underlying.upper()}{expiration.strftime('%y%m%d')}{cp}{strike_int:08d}"


def parse_osi_symbol(symbol: str) -> tuple[str, date, str, float]:
    """Parse OSI symbol into underlying, expiration, type, strike."""
    underlying = symbol[:3] if symbol.startswith("SPY") else symbol.split()[0]
    if not symbol.startswith("SPY") or len(symbol) < 15:
        raise ValueError(f"Unsupported OSI symbol format: {symbol}")
    exp_str = symbol[3:9]
    cp = symbol[9]
    strike_raw = symbol[10:]
    expiration = date(2000 + int(exp_str[0:2]), int(exp_str[2:4]), int(exp_str[4:6]))
    option_type = "call" if cp.upper() == "C" else "put"
    strike = int(strike_raw) / 1000.0
    return underlying, expiration, option_type, strike


def metadata_from_osi(
    osi_symbol: str,
    *,
    instrument_id: str | None = None,
) -> OptionContractMetadata:
    underlying, expiration, option_type, strike = parse_osi_symbol(osi_symbol)
    return OptionContractMetadata(
        underlying_symbol=underlying,
        expiration=expiration,
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        contract_id=osi_symbol,
        instrument_id=instrument_id,
        source="webull_opra",
    )


def candidate_strikes_around(underlying_price: float, step: float = 1.0) -> list[float]:
    base = round(underlying_price)
    return [base - step, base, base + step]


def build_atm_candidate_symbols(
    underlying: str,
    expiration: date,
    underlying_price: float,
) -> list[OptionContractMetadata]:
    """Construct ATM/near-ATM OSI symbols when chain API is unverified."""
    contracts: list[OptionContractMetadata] = []
    for strike in candidate_strikes_around(underlying_price):
        for opt_type in ("call", "put"):
            osi = build_osi_symbol(underlying, expiration, strike, opt_type)
            contracts.append(metadata_from_osi(osi))
    return contracts
