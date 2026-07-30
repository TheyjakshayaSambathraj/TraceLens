"""PII detection and redaction service.

Architecture
------------
PIIRedactor is the single, authoritative PII redaction component.
No regex patterns for PII detection are scattered elsewhere in the codebase.

The redactor is designed to:
1. Detect common PII categories using deterministic patterns.
2. Replace detected PII with typed placeholder tokens.
3. Preserve all non-PII content — including numeric decision evidence.
4. Handle strings, dicts, lists, and arbitrary nested structures.
5. Be architecturally replaceable with a more sophisticated recognizer
   (e.g., Microsoft Presidio) without touching call sites.

Redaction targets
-----------------
EMAIL            → [REDACTED_EMAIL]
PHONE            → [REDACTED_PHONE]
IP_ADDRESS       → [REDACTED_IP]
GOV_ID           → [REDACTED_GOV_ID]   (SSN, Aadhaar, etc.)
PERSON_NAME      → [REDACTED_NAME]     (heuristic: Title Case two-word patterns)

Non-targets (preserved)
-----------------------
Numeric values (leave_balance, duration, counts)
Boolean values
Policy references (Section 2.3, etc.)
Employee IDs (EMP-001) — these are organizational identifiers, not PII
already-redacted placeholders — idempotent

Design decisions
----------------
- We do NOT use Presidio to avoid a heavy dependency chain.
  The architecture allows a future drop-in replacement.
- Name detection uses a conservative heuristic to minimize false positives.
  It only fires on two-or-more capitalised words that aren't known safe
  patterns (employee IDs, policy sections, department names).
- The redactor is stateless and thread-safe.
- The singleton is a module-level instance cached by get_redactor().
"""

from __future__ import annotations

import re
import structlog
from typing import Any

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Placeholder tokens (consistent across the entire system)
# ---------------------------------------------------------------------------

REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
REDACTED_IP = "[REDACTED_IP]"
REDACTED_GOV_ID = "[REDACTED_GOV_ID]"
REDACTED_NAME = "[REDACTED_NAME]"

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Email: RFC-5321 simplified
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Phone: international and local formats
# +91-9876543210, +1 (555) 123-4567, 07911 123456, 555-123-4567
_PHONE_RE = re.compile(
    r"""
    (?:\+\d{1,3}[\s\-]?)?       # optional country code
    (?:\(?\d{2,4}\)?[\s\-.])?   # optional area code
    \d{3,5}[\s\-.]?\d{3,5}      # main number
    (?:[\s\-.]?\d{2,4})?        # optional extension
    """,
    re.VERBOSE,
)

# IPv4
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 (simplified)
_IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
)

# Government-style identifiers
# US SSN: 123-45-6789
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Indian Aadhaar: 1234 5678 9012 or 123456789012
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

# Person name heuristic:
# Two or more consecutive Title-Case words, not matching known safe patterns.
_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{1,20})(?:\s+([A-Z][a-z]{1,20})){1,3}\b"
)

# Words that look like Title-Case names but are NOT personal names.
# When any word in a candidate match is in this set, the whole match is kept.
_SAFE_WORDS = {
    # English sentence starters and grammar words
    "The", "A", "An", "This", "That", "These", "Those",
    "For", "Of", "In", "On", "At", "By", "To", "From",
    "And", "Or", "But", "So", "Yet", "Nor",
    "See", "Per", "As", "Up", "Out", "With", "Without",
    "Under", "Over", "Into", "Upon", "Via",
    # Document/policy structure
    "Section", "Article", "Clause", "Appendix", "Annex",
    "Chapter", "Part", "Schedule", "Exhibit", "Policy",
    "Paragraph", "Subsection", "Procedure", "Guideline",
    # Date/time words
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    # Organizational identifiers
    "Engineering", "Finance", "Sales", "Marketing", "Operations",
    "Hr", "Legal", "Product", "Design", "Research",
    "Management", "Executive", "Department", "Team", "Division",
    # Job levels and titles
    "Senior", "Junior", "Standard", "Manager", "Director",
    "Officer", "Lead", "Staff", "Principal", "Associate",
    # Decision / status words
    "Approved", "Rejected", "Pending", "Review", "Active",
    "Completed", "Failed", "Status", "Decision",
    # Company suffixes
    "Corp", "Ltd", "Inc", "Llc", "Plc", "Group",
    # Application-specific
    "TraceLens", "Employee", "Leave", "Request", "Balance",
    "Annual", "Sick", "Casual", "Maternity", "Paternity",
    # Common verbs/predicates that appear Title-Cased after sentence start
    "Has", "Have", "Had", "Was", "Were", "Is", "Are", "Be",
    "Can", "Cannot", "Will", "Would", "Should", "Shall",
    "Does", "Did", "Do", "Not", "No", "Yes",
    # REDACTED tokens (never treat as names)
    "Redacted",
}

# Also compile a prefix check for EMP-style identifiers
_SAFE_PREFIX_RE = re.compile(r"^(?:EMP|USR|USER|ORG|DEPT|REF|TL)-", re.IGNORECASE)

# Already-redacted placeholder — do not double-redact
_ALREADY_REDACTED_RE = re.compile(r"\[REDACTED_[A-Z_]+\]")

# Pre-computed lowercase version for case-insensitive word lookups
_SAFE_WORDS_LOWER: frozenset[str] = frozenset(w.lower() for w in _SAFE_WORDS)



class PIIRedactor:
    """Stateless PII detection and redaction engine.

    Applies a multi-pass redaction pipeline to any input value.
    Handles strings, dicts, lists, and nested combinations thereof.

    The redactor is designed to be conservative on false positives:
    better to retain a borderline value than to destroy useful audit
    evidence that has no PII.
    """

    def redact(self, value: Any) -> Any:
        """Redact PII from any value type.

        Args:
            value: Input to redact. May be str, int, float, bool,
                   dict, list, or None.

        Returns:
            Redacted value of the same structural type.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            # bool check BEFORE int (bool is subclass of int)
            return value
        if isinstance(value, (int, float)):
            # Numeric values are never PII — preserve leave_balance etc.
            return value
        if isinstance(value, str):
            return self._redact_string(value)
        if isinstance(value, dict):
            return {k: self.redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            redacted = [self.redact(item) for item in value]
            return type(value)(redacted)
        # Fallback: convert to string and redact
        return self._redact_string(str(value))

    def _redact_string(self, text: str) -> str:
        """Apply all PII patterns to a string value.

        Applies patterns in priority order, highest-specificity first.

        Args:
            text: Input string to redact.

        Returns:
            String with PII replaced by typed placeholder tokens.
        """
        if not text or _ALREADY_REDACTED_RE.fullmatch(text.strip()):
            return text

        # Track whether any redaction occurred
        original = text

        # 1. Emails (high specificity — apply first)
        text = _EMAIL_RE.sub(REDACTED_EMAIL, text)

        # 2. Government IDs (before phone, as SSN looks like phone)
        text = _SSN_RE.sub(REDACTED_GOV_ID, text)
        text = _AADHAAR_RE.sub(REDACTED_GOV_ID, text)

        # 3. IP addresses (before phone — digits could overlap)
        text = _IPV4_RE.sub(REDACTED_IP, text)
        text = _IPV6_RE.sub(REDACTED_IP, text)

        # 4. Phone numbers
        text = self._redact_phones(text)

        # 5. Person names (heuristic, lowest priority)
        text = self._redact_names(text)

        if text != original:
            logger.debug("pii_redacted", original_length=len(original))

        return text

    def _redact_phones(self, text: str) -> str:
        """Apply phone number redaction with length guard.

        Phone patterns can match short digit sequences that aren't phones
        (e.g., "12 days" or "20 balance"). We apply a length heuristic to
        avoid false positives on short numeric strings.

        Args:
            text: Input text.

        Returns:
            Text with phone numbers replaced.
        """
        def _replace_phone(match: re.Match) -> str:
            matched = match.group(0).strip()
            # Only treat as phone if it contains 7+ digits
            digits = re.sub(r"\D", "", matched)
            if len(digits) >= 7:
                return REDACTED_PHONE
            return matched

        return _PHONE_RE.sub(_replace_phone, text)

    def _redact_names(self, text: str) -> str:
        """Apply conservative person-name heuristic.

        Matches Title Case word pairs/triples that are not known-safe
        organizational terms or already-redacted placeholders.

        Args:
            text: Input text.

        Returns:
            Text with detected names replaced.
        """
        def _replace_name(match: re.Match) -> str:
            candidate = match.group(0)

            # Don't touch already-redacted content
            if _ALREADY_REDACTED_RE.search(candidate):
                return candidate

            # Check each word: if ANY word is a known safe word → keep it
            words = candidate.split()
            for word in words:
                # Check the safe-word set (case-insensitive, O(1) lookup)
                if word.lower() in _SAFE_WORDS_LOWER:
                    return candidate
                # Check organizational ID prefixes (EMP-001, USER-001, etc.)
                if _SAFE_PREFIX_RE.match(word):
                    return candidate

            # Two or more genuine Title-Case words — treat as a name
            return REDACTED_NAME

        return _NAME_RE.sub(_replace_name, text)

    def is_pii_free(self, value: Any) -> bool:
        """Check whether a value appears to be free of PII.

        Used for post-redaction verification in tests and monitoring.

        Args:
            value: Value to inspect.

        Returns:
            True if no PII patterns detected, False otherwise.
        """
        text = str(value) if not isinstance(value, str) else value
        if _EMAIL_RE.search(text):
            return False
        if _SSN_RE.search(text):
            return False
        # Only flag phone if it has 7+ digits
        for m in _PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) >= 7:
                return False
        return True


# Module-level singleton
_redactor_instance: PIIRedactor | None = None


def get_redactor() -> PIIRedactor:
    """Return the module-level PIIRedactor singleton.

    Thread-safe: PIIRedactor is stateless so sharing it across threads
    is safe without locking.

    Returns:
        PIIRedactor instance.
    """
    global _redactor_instance
    if _redactor_instance is None:
        _redactor_instance = PIIRedactor()
        logger.info("pii_redactor_initialized")
    return _redactor_instance
