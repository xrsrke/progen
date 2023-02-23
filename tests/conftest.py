import pytest
from progen.utils import yaml2dict

@pytest.fixture
def default_config():
    return yaml2dict('./configs/default.yaml')