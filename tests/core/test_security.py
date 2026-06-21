"""ตรวจ PII redaction — Thai ID checksum, email, phone, credit card"""
from core.security import _is_valid_thai_id, anonymize_pii


class TestThaiIDChecksum:
    def test_valid_known_id(self):
        # digits = "112345678912X"
        # sum = 13+12+22+30+36+40+42+42+40+36+3+4 = 320
        # check = (11 - 320%11) % 10 = (11 - 1) % 10 = 0
        assert _is_valid_thai_id("1123456789120") is True

    def test_invalid_checksum(self):
        assert _is_valid_thai_id("1234567890123") is False

    def test_wrong_length(self):
        assert _is_valid_thai_id("12345") is False

    def test_non_digit(self):
        assert _is_valid_thai_id("1abc456789012") is False


class TestAnonymize:
    def test_email_redacted(self):
        text = "ส่งเมลถึง user@example.com แล้วนะ"
        out, found = anonymize_pii(text)
        assert "[REDACTED_EMAIL]" in out
        assert "user@example.com" not in out
        assert found is True

    def test_thai_mobile_redacted(self):
        text = "เบอร์ 081-234-5678 ของฉัน"
        out, found = anonymize_pii(text)
        assert "[REDACTED_PHONE]" in out
        assert found is True

    def test_credit_card_redacted(self):
        text = "บัตร 4111 1111 1111 1111"
        out, found = anonymize_pii(text)
        assert "[REDACTED_CREDIT_CARD]" in out
        assert found is True

    def test_no_pii(self):
        text = "ตลาดวันนี้เป็นยังไงบ้าง"
        out, found = anonymize_pii(text)
        assert out == text
        assert found is False

    def test_valid_thai_id_redacted(self):
        text = "เลข 1-1234-56789-12-0 ของฉัน"
        out, found = anonymize_pii(text)
        assert "[REDACTED_THAI_ID]" in out
        assert found is True

    def test_invalid_thai_id_not_redacted(self):
        # 13 digits แต่ checksum ผิด → ไม่ใช่เลข ปชช จริง
        text = "เลข 1-2345-67890-12-3 ของฉัน"
        out, _ = anonymize_pii(text)
        assert "[REDACTED_THAI_ID]" not in out
