import unittest
from unittest.mock import patch, MagicMock

import torch

from predict import normalize_sql, predict, get_model


class TestNormalizeSql(unittest.TestCase):

    def test_strip_whitespace(self):
        assert normalize_sql("  SELECT id  ") == "SELECT id"

    def test_collapse_multiple_spaces(self):
        assert normalize_sql("SELECT  id   FROM   t") == "SELECT id FROM t"

    def test_double_quotes_to_single(self):
        assert normalize_sql('SELECT "name" FROM t') == "SELECT 'name' FROM t"

    def test_tabs_and_newlines(self):
        assert normalize_sql("SELECT\tid\nFROM\tt") == "SELECT id FROM t"

    def test_empty_string(self):
        assert normalize_sql("") == ""

    def test_idempotent(self):
        raw = '  SELECT  "col1" ,  col2  FROM  Users  '
        result = normalize_sql(raw)
        assert normalize_sql(result) == result

    def test_lowercases_identifiers(self):
        assert normalize_sql("SELECT * FROM Users") == "SELECT * FROM users"

    def test_keywords_uppercased(self):
        assert normalize_sql("select id from t where x = 1") == "SELECT id FROM t WHERE x = 1"

    def test_invalid_sql_still_normalizes(self):
        result = normalize_sql("NOT VALID SQL !!!")
        assert result == "not valid sql !!!"


class TestGetModel(unittest.TestCase):

    @patch("predict.T5ForConditionalGeneration")
    @patch("predict.T5Tokenizer")
    @patch("os.path.exists", return_value=True)
    def test_lazy_loads_once(self, mock_exists, mock_tok_cls, mock_model_cls):
        import predict
        predict.tokenizer = None
        predict.model = None

        get_model()
        get_model()

        mock_tok_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()

        predict.tokenizer = None
        predict.model = None

    @patch("predict.T5ForConditionalGeneration")
    @patch("predict.T5Tokenizer")
    @patch("os.path.exists", return_value=False)
    def test_falls_back_to_huggingface(self, mock_exists, mock_tok_cls, mock_model_cls):
        import predict
        predict.tokenizer = None
        predict.model = None

        get_model()

        mock_tok_cls.from_pretrained.assert_called_once_with(predict.HF_MODEL_ID)
        mock_model_cls.from_pretrained.assert_called_once_with(predict.HF_MODEL_ID)

        predict.tokenizer = None
        predict.model = None


class TestPredict(unittest.TestCase):

    def setUp(self):
        import predict as mod
        self.mod = mod
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        self.mod.tokenizer = self.mock_tokenizer
        self.mod.model = self.mock_model

        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        self.mock_model.generate.return_value = torch.tensor([[4, 5, 6]])
        self.mock_tokenizer.decode.return_value = "SELECT id FROM users"

    def tearDown(self):
        self.mod.tokenizer = None
        self.mod.model = None

    def test_returns_decoded_string(self):
        result = predict("Show all user ids", db_id="test_db")
        assert result == "SELECT id FROM users"

    def test_uses_prompt_template(self):
        predict("my question", db_id="my_db", schema="t1(a, b)")
        input_text = self.mock_tokenizer.call_args[0][0]
        assert "my_db" in input_text
        assert "my question" in input_text
        assert "t1(a, b)" in input_text

    def test_default_db_id_and_schema(self):
        predict("some question")
        input_text = self.mock_tokenizer.call_args[0][0]
        assert "unknown" in input_text

    def test_calls_beam_search(self):
        predict("q")
        gen_kwargs = self.mock_model.generate.call_args[1]
        assert gen_kwargs["num_beams"] == 5


if __name__ == "__main__":
    unittest.main()
