"""Unit tests for PIIRedactor.

Tests cover:
- Email detection and replacement
- Phone number detection with length guard
- IP address (v4 and v6) detection
- Government ID (SSN, Aadhaar) detection
- Name heuristic detection and safe-word bypass
- Nested dict/list redaction
- Numeric preservation (CRITICAL: leave_balance must survive)
- Boolean preservation
- Idempotency on already-redacted content
- Multiple PII types in single string
- is_pii_free() verification helper
"""

from __future__ import annotations

import pytest

from app.privacy.redactor import (
    PIIRedactor,
    REDACTED_EMAIL,
    REDACTED_PHONE,
    REDACTED_IP,
    REDACTED_GOV_ID,
    REDACTED_NAME,
    get_redactor,
)


@pytest.fixture
def redactor() -> PIIRedactor:
    """Return a fresh PIIRedactor for each test."""
    return PIIRedactor()


class TestEmailRedaction:
    """Email address detection and replacement."""

    def test_simple_email(self, redactor: PIIRedactor):
        result = redactor.redact("Contact john.doe@company.com for info")
        assert REDACTED_EMAIL in result
        assert "john.doe@company.com" not in result

    def test_email_at_start(self, redactor: PIIRedactor):
        result = redactor.redact("hr@corp.org approved the request")
        assert REDACTED_EMAIL in result
        assert "hr@corp.org" not in result

    def test_email_alone(self, redactor: PIIRedactor):
        assert redactor.redact("user@example.com") == REDACTED_EMAIL

    def test_multiple_emails(self, redactor: PIIRedactor):
        result = redactor.redact("alice@a.com and bob@b.com")
        assert result.count(REDACTED_EMAIL) == 2
        assert "@" not in result

    def test_no_false_positive_on_policy_section(self, redactor: PIIRedactor):
        """Section 2.3 should NOT trigger email redaction."""
        result = redactor.redact("See Section 2.3 for details")
        assert REDACTED_EMAIL not in result
        assert "Section 2.3" in result


class TestPhoneRedaction:
    """Phone number detection with length guard."""

    def test_international_phone(self, redactor: PIIRedactor):
        result = redactor.redact("Call +91-9876543210 for HR")
        assert REDACTED_PHONE in result
        assert "9876543210" not in result

    def test_us_phone_format(self, redactor: PIIRedactor):
        result = redactor.redact("Phone: 555-867-5309")
        assert REDACTED_PHONE in result

    def test_short_number_not_redacted(self, redactor: PIIRedactor):
        """'12 days' should NOT be treated as a phone number."""
        result = redactor.redact("Employee has 12 days remaining")
        assert REDACTED_PHONE not in result
        assert "12" in result

    def test_leave_balance_preserved(self, redactor: PIIRedactor):
        """leave_balance: 20 must survive redaction unchanged."""
        result = redactor.redact({"leave_balance": 20, "status": "ACTIVE"})
        assert result["leave_balance"] == 20

    def test_duration_days_not_redacted(self, redactor: PIIRedactor):
        """'15 consecutive days' must not be redacted."""
        result = redactor.redact("Can employee take 15 consecutive days?")
        assert "15" in result
        assert REDACTED_PHONE not in result


class TestIPRedaction:
    """IP address detection."""

    def test_ipv4_address(self, redactor: PIIRedactor):
        result = redactor.redact("Request from 192.168.1.42")
        assert REDACTED_IP in result
        assert "192.168.1.42" not in result

    def test_ipv4_public(self, redactor: PIIRedactor):
        result = redactor.redact("Origin: 203.0.113.1")
        assert REDACTED_IP in result

    def test_ipv4_in_dict(self, redactor: PIIRedactor):
        data = {"client_ip": "10.0.0.1", "user": "EMP-001"}
        result = redactor.redact(data)
        assert REDACTED_IP in result["client_ip"]
        assert result["user"] == "EMP-001"

    def test_no_false_positive_version_string(self, redactor: PIIRedactor):
        """Version strings like 1.0.0 should not be treated as IPs."""
        # Only 4-octet patterns match; 3-segment version strings don't
        result = redactor.redact("App version 1.0.0")
        # 1.0.0 is only 3 segments, not a valid IPv4
        assert "1.0.0" in result


class TestGovernmentIDRedaction:
    """Government-issued ID detection."""

    def test_us_ssn(self, redactor: PIIRedactor):
        result = redactor.redact("SSN: 123-45-6789")
        assert REDACTED_GOV_ID in result
        assert "123-45-6789" not in result

    def test_aadhaar_with_spaces(self, redactor: PIIRedactor):
        result = redactor.redact("Aadhaar: 1234 5678 9012")
        assert REDACTED_GOV_ID in result
        assert "1234 5678 9012" not in result


class TestNameRedaction:
    """Person name heuristic detection."""

    def test_full_name_redacted(self, redactor: PIIRedactor):
        result = redactor.redact("Approved by John Smith from HR")
        assert REDACTED_NAME in result
        assert "John Smith" not in result

    def test_employee_id_not_a_name(self, redactor: PIIRedactor):
        """EMP-001 must not trigger name redaction."""
        result = redactor.redact("Employee EMP-001 requested leave")
        assert REDACTED_NAME not in result
        assert "EMP-001" in result

    def test_department_name_not_redacted(self, redactor: PIIRedactor):
        """'Engineering' department should not be treated as a name."""
        result = redactor.redact("Employee is in Engineering")
        assert "Engineering" in result

    def test_decision_words_not_names(self, redactor: PIIRedactor):
        """APPROVED/REJECTED status words must not be redacted as names."""
        result = redactor.redact("Status: Approved")
        assert "Approved" in result or REDACTED_NAME not in result

    def test_policy_section_not_a_name(self, redactor: PIIRedactor):
        result = redactor.redact("See Section 2 for details")
        assert REDACTED_NAME not in result


class TestNestedStructures:
    """PII redaction in nested dicts and lists."""

    def test_dict_values_redacted(self, redactor: PIIRedactor):
        data = {
            "contact_email": "test@example.com",
            "department": "Engineering",
            "leave_balance": 15,
        }
        result = redactor.redact(data)
        assert REDACTED_EMAIL in result["contact_email"]
        assert result["department"] == "Engineering"
        assert result["leave_balance"] == 15  # numeric preserved

    def test_nested_dict(self, redactor: PIIRedactor):
        data = {
            "employee": {
                "email": "emp@corp.com",
                "id": "EMP-002",
            },
            "balance": 20,
        }
        result = redactor.redact(data)
        assert REDACTED_EMAIL in result["employee"]["email"]
        assert result["employee"]["id"] == "EMP-002"
        assert result["balance"] == 20

    def test_list_redaction(self, redactor: PIIRedactor):
        data = ["contact@domain.com", "EMP-001", "normal text"]
        result = redactor.redact(data)
        assert REDACTED_EMAIL in result[0]
        assert result[1] == "EMP-001"
        assert result[2] == "normal text"

    def test_mixed_list_in_dict(self, redactor: PIIRedactor):
        data = {
            "evidence": [
                "Employee balance: 12 days",
                "Email on file: person@hr.com",
            ]
        }
        result = redactor.redact(data)
        assert "12" in result["evidence"][0]  # numeric context preserved
        assert REDACTED_EMAIL in result["evidence"][1]

    def test_none_preserved(self, redactor: PIIRedactor):
        assert redactor.redact(None) is None

    def test_boolean_preserved(self, redactor: PIIRedactor):
        assert redactor.redact(True) is True
        assert redactor.redact(False) is False

    def test_integer_preserved(self, redactor: PIIRedactor):
        assert redactor.redact(42) == 42

    def test_float_preserved(self, redactor: PIIRedactor):
        assert redactor.redact(3.14) == 3.14


class TestIdempotency:
    """Already-redacted placeholders must not be double-redacted."""

    def test_already_redacted_email(self, redactor: PIIRedactor):
        result = redactor.redact(REDACTED_EMAIL)
        assert result == REDACTED_EMAIL

    def test_already_redacted_phone(self, redactor: PIIRedactor):
        result = redactor.redact(REDACTED_PHONE)
        assert result == REDACTED_PHONE


class TestMultiplePII:
    """Multiple PII types in a single string."""

    def test_email_and_phone_in_one_string(self, redactor: PIIRedactor):
        text = "Send to alice@example.com or call +44-7700-900123"
        result = redactor.redact(text)
        assert REDACTED_EMAIL in result
        assert REDACTED_PHONE in result
        assert "@" not in result

    def test_email_and_name(self, redactor: PIIRedactor):
        text = "Manager Alice Brown sent hr@example.com"
        result = redactor.redact(text)
        assert REDACTED_EMAIL in result
        # Name heuristic may or may not catch this depending on context


class TestPIIFreeCheck:
    """is_pii_free() verification helper."""

    def test_clean_text_is_pii_free(self, redactor: PIIRedactor):
        assert redactor.is_pii_free("Employee has 20 leave days") is True

    def test_email_is_not_pii_free(self, redactor: PIIRedactor):
        assert redactor.is_pii_free("contact@example.com") is False

    def test_ssn_is_not_pii_free(self, redactor: PIIRedactor):
        assert redactor.is_pii_free("SSN: 123-45-6789") is False


class TestSingleton:
    """Singleton factory test."""

    def test_get_redactor_same_instance(self):
        r1 = get_redactor()
        r2 = get_redactor()
        assert r1 is r2
