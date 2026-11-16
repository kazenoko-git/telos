"""Typed document scoring. Status/value, extraction and grounding stay separate."""
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import re

STATUSES = ('ANSWER', 'ABSENT', 'UNKNOWN', 'CONFLICT', 'AMBIGUOUS')
DATE_FORMATS = ('%Y-%m-%d', '%d %B %Y', '%B %d, %Y', '%B %d %Y', '%d %b %Y', '%b %d, %Y', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%d-%m-%Y')


def equal(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(equal(actual.get(k), v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
    try:
        return Decimal(str(actual)) == Decimal(str(expected))
    except InvalidOperation:
        return str(actual) == str(expected)


def normalize_value(kind, value):
    if value is None:
        return None
    if kind == 'money':
        if isinstance(value, dict):
            amount = value.get('amount')
            if amount is None:
                return value
            amount = str(amount)
            for char in '$€£¥,':
                amount = amount.replace(char, '')
            currency = value.get('currency')
            return {'amount': amount.strip(), 'currency': str(currency).upper().strip() if currency else None}
        if isinstance(value, (int, float)):
            return {'amount': str(value), 'currency': None}
        if isinstance(value, str):
            match = re.search(r'\b(USD|CAD|EUR|GBP|JPY|AUD|CHF|CNY|INR|NZD|SEK)\b', value.upper())
            currency = match.group(1) if match else None
            if not currency:
                for symbol, code in [('C$', 'CAD'), ('A$', 'AUD'), ('€', 'EUR'), ('£', 'GBP'), ('¥', 'JPY'), ('$', 'USD')]:
                    if symbol in value:
                        currency = code
                        break
            match = re.search(r'[-+]?\d*\.?\d+', value.replace(',', ''))
            if match:
                return {'amount': match.group(), 'currency': currency}
    if kind == 'date' and isinstance(value, str):
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(value.strip(), fmt).date().isoformat()
            except ValueError:
                pass
    if kind in ('number', 'pct') and isinstance(value, str):
        text = value.strip().rstrip('%').strip()
        if ',' in text and '.' not in text:
            text = text.replace(',', '')
        try:
            return str(Decimal(text))
        except InvalidOperation:
            pass
    if kind == 'list' and isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass
    return value


def parse_object(raw):
    if not isinstance(raw, str):
        raise ValueError('Generation must be a JSON string')
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)[:-3].strip()
    data = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'Non-finite JSON number: {x}')))
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object')
    return data


def parse_direct(raw, kind):
    data = parse_object(raw)
    return {**data, 'status': str(data.get('status', 'ANSWER')).upper(), 'value': normalize_value(kind, data.get('value'))}


def parse_fields(raw, names):
    data = parse_object(raw)
    fields = {}
    for name in names:
        value = data.get(name, [])
        if value is None or value == '':
            value = []
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(v, (str, int, float)) or isinstance(v, bool) for v in value):
            raise ValueError(f'Field {name} must be a literal string or list of literals')
        fields[name] = [str(v) for v in value if str(v)]
    return fields


def outcome(case, response):
    status = response.get('status')
    error = response.get('error')
    if status == 'MALFORMED':
        return 'malformed_response'
    if status == 'CONTRACT_ERROR':
        return 'contract_error'
    if error or status in ('ERROR', 'RUNTIME_ERROR'):
        code = error.get('http_status') if isinstance(error, dict) else None
        return 'rate_limit_error' if code == 429 else 'runtime_error'
    if status not in STATUSES:
        return 'unsupported_output'
    accepted = case.get('statuses', ['ANSWER'])
    if 'expected' in case:
        if status not in accepted:
            return 'incorrect_abstention'
        return 'correct_answer' if equal(response.get('value'), case['expected']) else 'incorrect_answer'
    return 'correct_abstention' if status in accepted else 'incorrect_answer' if status == 'ANSWER' else 'incorrect_abstention'


def evidence_info(document, response):
    evidence = response.get('evidence') or []
    if not isinstance(evidence, list):
        evidence = [evidence]
    rows = []
    for entry in evidence:
        quote = entry if isinstance(entry, str) else entry.get('quote') if isinstance(entry, dict) else None
        start = entry.get('start') if isinstance(entry, dict) else None
        end = entry.get('end') if isinstance(entry, dict) else None
        native = isinstance(start, int) and isinstance(end, int)
        valid = native and isinstance(quote, str) and 0 <= start < end <= len(document) and document[start:end] == quote
        found = document.find(quote) if isinstance(quote, str) and quote else -1
        rows.append({'quote': quote, 'start': start if valid else found,
                     'end': end if valid else found + len(quote) if found >= 0 else -1,
                     'source': 'model' if valid else 'quote_match', 'model_offsets_supplied': native,
                     'model_offsets_valid': bool(valid), 'literal_match': found >= 0})
    return {'spans': rows, 'has_evidence': bool(rows),
            'all_quotes_match': all(r['literal_match'] for r in rows) if rows else None,
            'model_offsets_supplied': sum(r['model_offsets_supplied'] for r in rows),
            'valid_model_offsets': sum(r['model_offsets_valid'] for r in rows)}
