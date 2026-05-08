import sys
import unittest
from unittest.mock import patch, MagicMock
import re

import torch


def _import_predict():
    """Import predict module with mocked model loading."""
    if "predict" in sys.modules:
        return sys.modules["predict"]
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    with patch("os.path.exists", return_value=True), \
         patch("transformers.T5Tokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.T5ForConditionalGeneration.from_pretrained", return_value=mock_model):
        import predict
    return predict


predict_module = _import_predict()


class TestNormalizeSql(unittest.TestCase):

    def test_lowercase(self):
        assert predict_module.normalize_sql("SELECT * FROM Users") == "select * from users"

    def test_strip_whitespace(self):
        assert predict_module.normalize_sql("  SELECT id  ") == "select id"

    def test_collapse_multiple_spaces(self):
        assert predict_module.normalize_sql("SELECT  id   FROM   t") == "select id from t"

    def test_double_quotes_to_single(self):
        assert predict_module.normalize_sql('SELECT "name" FROM t') == "select 'name' from t"

    def test_remove_space_before_comma(self):
        assert predict_module.normalize_sql("SELECT a , b FROM t") == "select a,b from t"

    def test_remove_space_after_comma(self):
        assert predict_module.normalize_sql("SELECT a, b, c FROM t") == "select a,b,c from t"

    def test_combined_normalization(self):
        raw = '  SELECT  "col1" , col2,  col3  FROM  Users  '
        expected = "select 'col1',col2,col3 from users"
        assert predict_module.normalize_sql(raw) == expected

    def test_already_normalized(self):
        sql = "select id from t where x = 1"
        assert predict_module.normalize_sql(sql) == sql

    def test_tabs_and_newlines(self):
        assert predict_module.normalize_sql("SELECT\tid\nFROM\tt") == "select id from t"

    def test_empty_string(self):
        assert predict_module.normalize_sql("") == ""


class TestPredict(unittest.TestCase):

    def test_predict_returns_decoded_string(self):
        predict_module.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        predict_module.model.generate.return_value = torch.tensor([[4, 5, 6]])
        predict_module.tokenizer.decode.return_value = "SELECT id FROM users"

        result = predict_module.predict("Show all user ids", db_id="test_db")
        assert result == "SELECT id FROM users"

    def test_predict_uses_prompt_template(self):
        predict_module.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        predict_module.model.generate.return_value = torch.tensor([[1]])
        predict_module.tokenizer.decode.return_value = ""

        predict_module.predict("my question", db_id="my_db")

        input_text = predict_module.tokenizer.call_args[0][0]
        assert "my_db" in input_text
        assert "my question" in input_text

    def test_predict_default_db_id(self):
        predict_module.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        predict_module.model.generate.return_value = torch.tensor([[1]])
        predict_module.tokenizer.decode.return_value = ""

        predict_module.predict("some question")

        input_text = predict_module.tokenizer.call_args[0][0]
        assert "unknown" in input_text

    def test_predict_calls_beam_search(self):
        predict_module.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        predict_module.model.generate.return_value = torch.tensor([[1]])
        predict_module.tokenizer.decode.return_value = ""

        predict_module.predict("q")

        gen_kwargs = predict_module.model.generate.call_args[1]
        assert gen_kwargs["num_beams"] == 5


if __name__ == "__main__":
    unittest.main()
