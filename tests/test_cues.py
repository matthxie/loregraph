from __future__ import annotations

from kg.cues import cue_kinds, has_cue


def test_quantity_cue_currency_symbol():
    assert has_cue("I got a Gucci handbag for $1,200.")
    assert "quantity" in cue_kinds("earning $225 at the market")
    assert has_cue("it only cost $7.5 each")


def test_quantity_cue_money_words():
    assert has_cue("I paid 20 bucks for it")
    assert has_cue("that's about 1200 dollars total")
    assert has_cue("cost me 50 cents")


def test_quantity_cue_measurement_units():
    assert has_cue("I bought 10 lbs of flour")
    assert has_cue("ran 3 miles this morning")
    assert has_cue("picked up 2 dozen eggs")


def test_quantity_cue_ignores_bare_numbers_dates_times_ordinals():
    # "in 1995" also fires _REL_DATE — assert quantity specifically stays quiet.
    assert "quantity" not in cue_kinds("in 1995 I moved to Boston")
    assert not has_cue("we met at 5pm")
    assert not has_cue("she came in 3rd place")
    # A bare count with no attached currency/unit is not a quantity cue by design —
    # escalation should be reserved for numerals that are actually summable amounts.
    assert not has_cue("3 of my friends came over")


def test_quantity_cue_does_not_fire_on_plain_text():
    # This sentence only has a rel_date cue ("yesterday") — quantity must stay quiet.
    assert cue_kinds("I had a great time at the party yesterday") == {"rel_date"}
