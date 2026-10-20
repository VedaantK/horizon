import pandas as pd
import pytest

from concordance.futures.continuous import Contract, build_continuous, point_changes

S = pd.DatetimeIndex(pd.to_datetime([
    "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-12",
    "2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19"]))


def bars(idx, closes):
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes},
                        index=pd.DatetimeIndex(idx))


def test_panama_two_contracts_hand_computed():
    # Front expires 03-19; roll 2 sessions before -> 03-17. Next trades 03-15 onward.
    front = Contract("H21", pd.Timestamp("2021-03-19"),
                     bars=bars(S, [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))
    nxt = Contract("M21", pd.Timestamp("2021-06-18"),
                   bars=bars(S[5:], [110, 111, 112, 113, 114]))
    series, table = build_continuous([nxt, front], sessions_before=2)
    # Roll on 03-17: gap = 112 - 107 = 5. Front segment shifted +5, next unadjusted.
    assert table.to_dict("records") == [{"roll_date": pd.Timestamp("2021-03-17"),
                                         "from_contract": "H21", "to_contract": "M21",
                                         "gap": 5.0}]
    assert series["close"].tolist() == [105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
    assert series["contract"].tolist() == ["H21"] * 8 + ["M21"] * 2
    assert series["close_unadj"].tolist()[7:9] == [107, 113]
    # Point change across the roll = next contract's own move (113 - 112), not the gap.
    assert point_changes(series).loc["2021-03-18"] == 1.0


def test_first_notice_forces_earlier_roll():
    # Physically settled: FND 03-17 -> must be out by 03-16; 1 session before -> 03-15.
    front = Contract("J21", pd.Timestamp("2021-03-26"), first_notice=pd.Timestamp("2021-03-17"),
                     bars=bars(S, range(100, 110)))
    nxt = Contract("K21", pd.Timestamp("2021-04-28"), bars=bars(S[3:], range(200, 207)))
    _, table = build_continuous([front, nxt], sessions_before=1)
    assert table["roll_date"].iloc[0] == pd.Timestamp("2021-03-15")


def test_roll_moves_earlier_never_later():
    # Next contract has no close on 03-17 (candidate) -> roll on 03-16, not 03-18.
    nb = bars(S[5:], [110, 111, 112, 113, 114]).drop(pd.Timestamp("2021-03-17"))
    front = Contract("H21", pd.Timestamp("2021-03-19"), bars=bars(S, range(100, 110)))
    nxt = Contract("M21", pd.Timestamp("2021-06-18"), bars=nb)
    _, table = build_continuous([front, nxt], sessions_before=2)
    assert table["roll_date"].iloc[0] == pd.Timestamp("2021-03-16")


def test_no_overlap_in_time_is_an_error():
    front = Contract("H21", pd.Timestamp("2021-03-19"), bars=bars(S, range(100, 110)))
    nxt = Contract("M21", pd.Timestamp("2021-06-18"), bars=bars(S[9:], [120]))
    with pytest.raises(ValueError, match="cannot roll in time"):
        build_continuous([front, nxt], sessions_before=2)


def test_roll_chain_works_before_1990():
    # Futures research starts 1990-06 with a 12-month warmup, so the first rolls are in 1989.
    from concordance.data.futures import _chain

    s = pd.DatetimeIndex(pd.to_datetime(["1989-06-12", "1989-06-13", "1989-06-14",
                                         "1989-06-15", "1989-06-16", "1989-06-19"]))
    front = Contract("M89", pd.Timestamp("1989-06-20"), bars=bars(s, range(100, 106)))
    nxt = Contract("U89", pd.Timestamp("1989-09-19"), bars=bars(s, range(110, 116)))
    chain, skipped = _chain([front, nxt], sessions_before=2)
    assert [c.id for c in chain] == ["M89", "U89"] and skipped == []
