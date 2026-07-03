import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require Qwen2.5-0.5B to be in the HF cache.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires a downloaded model — pass --run-integration to enable",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="pass --run-integration to run (needs model download)")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
