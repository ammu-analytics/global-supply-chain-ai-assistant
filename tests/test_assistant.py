import json

import pytest


def test_route_question(bot):
    ans = bot.ask("How much does Germany import from China in 2023?")
    assert ans.intent == "route_value"
    assert "175.6" in ans.text


def test_direction_is_resolved(bot):
    # only "China imports from Korea" exists in the data, whichever order the user writes
    a1 = bot.ask("korea to china trade in 2022")
    a2 = bot.ask("china korea trade in 2022")
    assert "Rep. of Korea → China" in a1.text and "Rep. of Korea → China" in a2.text


def test_total_mentions_cleaning(bot):
    ans = bot.ask("total trade value")
    assert "7.30 T" in ans.text
    assert "counted twice" in ans.text


def test_ranking(bot):
    assert "**China** is the biggest supplier" in bot.ask("who is the biggest supplier").text
    assert "Germany" in bot.ask("smallest importer").text


def test_unknown_year(bot):
    assert "no data for 2015" in bot.ask("total trade in 2015").text.lower()


def test_future_year_becomes_forecast(bot):
    ans = bot.ask("germany china trade in 2025")
    assert ans.intent == "forecast"
    assert "2025" in ans.text


def test_gibberish_is_not_guessed(bot):
    assert bot.ask("asdf qwer zxcv").intent == "unsure"


def test_empty_question(bot):
    assert "type a question" in bot.ask("   ").text


@pytest.mark.parametrize("q", [
    "hi", "show usa imports over the years", "which route grew the fastest",
    "forecast japan to usa", "supply chain risk", "any anomalies", "is the data clean",
])
def test_every_answer_is_json_serialisable(bot, q):
    json.dumps(bot.ask(q).to_dict())  # what the API sends back
