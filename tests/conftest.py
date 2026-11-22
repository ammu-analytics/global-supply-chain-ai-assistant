import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supplychain.assistant import SupplyChainAssistant  # noqa: E402
from supplychain.data import load_data  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return load_data()


@pytest.fixture(scope="session")
def bot(data):
    return SupplyChainAssistant(data=data)
