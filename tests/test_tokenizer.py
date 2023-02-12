import pytest
from tokenizers import Tokenizer

@pytest.fixture
def tokenizer_config():
    with open("./configs/tokenizer.json", "r") as f:
        return f.read()

@pytest.mark.parametrize(
    'protein, expected_ids',
    [
        ("2GFLPFRGADEGLAAREAA", [4, 11, 10, 15, 19, 10, 21, 11, 5, 8, 9, 11, 15, 5, 5, 21, 9, 5, 5]),
        ("MADDKTKIGTPDN", [16, 5, 8, 8, 14, 23, 14, 13, 11, 23, 19, 8, 17])
    ]
)
def test_tokenize_a_protein_sequence(tokenizer_config, protein, expected_ids):
    tokenizer = Tokenizer.from_str(tokenizer_config)

    ids = tokenizer.encode(protein).ids

    assert ids == expected_ids