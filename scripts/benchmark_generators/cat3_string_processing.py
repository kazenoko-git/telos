"""
Category 3: String & Text Processing (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_string_processing_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Longest Common Prefix
    tasks.append(create_task(
        name="longest_common_prefix",
        signature="def longest_common_prefix(strs: list[str]) -> str:",
        doc_desc="Finds the longest common prefix string amongst an array of strings.",
        doctests=[">>> longest_common_prefix(['flower', 'flow', 'flight'])", "'fl'"],
        hint="If strs is empty return ''. Use the first string as prefix; while any string does not start with prefix, shorten prefix by 1.",
        solution="    if not strs: return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix: return ''\n    return prefix",
        test="assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'\nassert longest_common_prefix(['dog', 'racecar', 'car']) == ''\nassert longest_common_prefix(['']) == ''\nassert longest_common_prefix(['alone']) == 'alone'\n"
    ))

    # 2. Is Anagram Case Insensitive
    tasks.append(create_task(
        name="is_anagram_case_insensitive",
        signature="def is_anagram_case_insensitive(s1: str, s2: str) -> bool:",
        doc_desc="Checks if two strings are anagrams of each other, ignoring case and whitespace.",
        doctests=[">>> is_anagram_case_insensitive('Listen', 'Silent')", "True"],
        hint="Filter out whitespace, convert to lower case, sort both character sequences, and compare for equality.",
        solution="    c1 = sorted(ch.lower() for ch in s1 if not ch.isspace())\n    c2 = sorted(ch.lower() for ch in s2 if not ch.isspace())\n    return c1 == c2",
        test="assert is_anagram_case_insensitive('Listen', 'Silent') is True\nassert is_anagram_case_insensitive('hello', 'world') is False\nassert is_anagram_case_insensitive('Clint Eastwood', 'Old West Action') is True\nassert is_anagram_case_insensitive('', '') is True\n"
    ))

    # 3. Caesar Cipher Encode
    tasks.append(create_task(
        name="caesar_cipher_encode",
        signature="def caesar_cipher_encode(text: str, shift: int) -> str:",
        doc_desc="Encodes ASCII letters with a caesar shift, wrapping around alphabets and preserving case and non-letters.",
        doctests=[">>> caesar_cipher_encode('abc', 3)", "'def'"],
        hint="For 'a'..'z', compute chr((ord(ch) - ord('a') + shift) % 26 + ord('a')). Do the same with 'A'..'Z'. Leave other chars untouched.",
        solution="    out = []\n    for ch in text:\n        if 'a' <= ch <= 'z':\n            out.append(chr((ord(ch) - ord('a') + shift) % 26 + ord('a')))\n        elif 'A' <= ch <= 'Z':\n            out.append(chr((ord(ch) - ord('A') + shift) % 26 + ord('A')))\n        else:\n            out.append(ch)\n    return ''.join(out)",
        test="assert caesar_cipher_encode('abc', 3) == 'def'\nassert caesar_cipher_encode('xyz', 2) == 'zab'\nassert caesar_cipher_encode('Hello, World!', 4) == 'Lipps, Asvph!'\nassert caesar_cipher_encode('abc', 0) == 'abc'\n"
    ))

    # 4. Vigenere Cipher Encode
    tasks.append(create_task(
        name="vigenere_cipher_encode",
        signature="def vigenere_cipher_encode(plaintext: str, key: str) -> str:",
        doc_desc="Encodes plaintext using the Vigenere cipher with alphabetic key, advancing key index only for alphabetic characters.",
        doctests=[">>> vigenere_cipher_encode('ATTACKATDAWN', 'LEMON')", "'LXFOPVEFRNHR'"],
        hint="Lowercase the key. Track key_idx. For each letter in plaintext, shift = ord(key[key_idx % len(key)]) - ord('a'), shift char and key_idx += 1.",
        solution="    if not key: return plaintext\n    key_clean = key.lower()\n    res = []\n    ki = 0\n    k_len = len(key_clean)\n    for ch in plaintext:\n        if 'a' <= ch <= 'z':\n            shift = ord(key_clean[ki % k_len]) - ord('a')\n            res.append(chr((ord(ch) - ord('a') + shift) % 26 + ord('a')))\n            ki += 1\n        elif 'A' <= ch <= 'Z':\n            shift = ord(key_clean[ki % k_len]) - ord('a')\n            res.append(chr((ord(ch) - ord('A') + shift) % 26 + ord('A')))\n            ki += 1\n        else:\n            res.append(ch)\n    return ''.join(res)",
        test="assert vigenere_cipher_encode('ATTACKATDAWN', 'LEMON') == 'LXFOPVEFRNHR'\nassert vigenere_cipher_encode('Hello World!', 'KEY') == 'Rijvs Uyvjn!'\n"
    ))

    # 5. ROT13 Encode
    tasks.append(create_task(
        name="rot13_encode",
        signature="def rot13_encode(text: str) -> str:",
        doc_desc="Applies ROT13 substitution cipher to text, preserving case and non-alphabetic characters.",
        doctests=[">>> rot13_encode('Hello, World!')", "'Uryyb, Jbeyq!'"],
        hint="Shift letters by 13 modulo 26 for both lowercase and uppercase letters.",
        solution="    out = []\n    for ch in text:\n        if 'a' <= ch <= 'z':\n            out.append(chr((ord(ch) - ord('a') + 13) % 26 + ord('a')))\n        elif 'A' <= ch <= 'Z':\n            out.append(chr((ord(ch) - ord('A') + 13) % 26 + ord('A')))\n        else:\n            out.append(ch)\n    return ''.join(out)",
        test="assert rot13_encode('Hello, World!') == 'Uryyb, Jbeyq!'\nassert rot13_encode('Uryyb, Jbeyq!') == 'Hello, World!'\nassert rot13_encode('1234') == '1234'\n"
    ))

    # 6. Run Length Encode
    tasks.append(create_task(
        name="run_length_encode",
        signature="def run_length_encode(s: str) -> str:",
        doc_desc="Compresses string s using run-length encoding (e.g. 'aaabbc' -> 'a3b2c1'). Returns '' for empty string.",
        doctests=[">>> run_length_encode('aaabbc')", "'a3b2c1'"],
        hint="Use itertools.groupby(s): for each char and group, append f'{char}{len(list(group))}'.",
        solution="    if not s: return ''\n    from itertools import groupby\n    return ''.join(f'{k}{len(list(g))}' for k, g in groupby(s))",
        test="assert run_length_encode('aaabbc') == 'a3b2c1'\nassert run_length_encode('abcd') == 'a1b1c1d1'\nassert run_length_encode('') == ''\nassert run_length_encode('a') == 'a1'\n"
    ))

    # 7. Run Length Decode
    tasks.append(create_task(
        name="run_length_decode",
        signature="def run_length_decode(s: str) -> str:",
        doc_desc="Decompresses run-length encoded string like 'a3b2c1' back into 'aaabbc'. Supports multi-digit counts.",
        doctests=[">>> run_length_decode('a3b2c1')", "'aaabbc'"],
        hint="Use regex re.findall(r'([a-zA-Z])(\\d+)', s) and multiply each char by int(count).",
        solution="    import re\n    matches = re.findall(r'([a-zA-Z])(\\d+)', s)\n    return ''.join(ch * int(count) for ch, count in matches)",
        test="assert run_length_decode('a3b2c1') == 'aaabbc'\nassert run_length_decode('a10') == 'aaaaaaaaaa'\nassert run_length_decode('') == ''\n"
    ))

    # 8. Snake to Camel Case
    tasks.append(create_task(
        name="snake_to_camel_case",
        signature="def snake_to_camel_case(s: str) -> str:",
        doc_desc="Converts snake_case identifier to lowerCamelCase.",
        doctests=[">>> snake_to_camel_case('hello_world_test')", "'helloWorldTest'"],
        hint="Split by '_'. Keep first component lowercase, capitalize all subsequent components, and join.",
        solution="    parts = [p for p in s.split('_') if p]\n    if not parts: return ''\n    return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])",
        test="assert snake_to_camel_case('hello_world_test') == 'helloWorldTest'\nassert snake_to_camel_case('single') == 'single'\nassert snake_to_camel_case('') == ''\nassert snake_to_camel_case('__leading_trail__') == 'leadingTrail'\n"
    ))

    # 9. Camel to Snake Case
    tasks.append(create_task(
        name="camel_to_snake_case",
        signature="def camel_to_snake_case(s: str) -> str:",
        doc_desc="Converts lowerCamelCase or UpperCamelCase string into lower_snake_case.",
        doctests=[">>> camel_to_snake_case('helloWorldTest')", "'hello_world_test'"],
        hint="Insert '_' before uppercase letters (except at index 0) and convert everything to lowercase.",
        solution="    res = []\n    for i, ch in enumerate(s):\n        if ch.isupper() and i > 0 and not s[i-1].isupper() and s[i-1] != '_':\n            res.append('_')\n        res.append(ch.lower())\n    return ''.join(res).strip('_')",
        test="assert camel_to_snake_case('helloWorldTest') == 'hello_world_test'\nassert camel_to_snake_case('HelloWorld') == 'hello_world'\nassert camel_to_snake_case('simple') == 'simple'\n"
    ))

    # 10. Kebab to Snake Case
    tasks.append(create_task(
        name="kebab_to_snake_case",
        signature="def kebab_to_snake_case(s: str) -> str:",
        doc_desc="Converts kebab-case string into snake_case, replacing hyphens with underscores.",
        doctests=[">>> kebab_to_snake_case('my-component-name')", "'my_component_name'"],
        hint="Replace all occurrences of '-' with '_'.",
        solution="    return s.replace('-', '_')",
        test="assert kebab_to_snake_case('my-component-name') == 'my_component_name'\nassert kebab_to_snake_case('test') == 'test'\nassert kebab_to_snake_case('') == ''\n"
    ))

    # 11. Count Vowels and Consonants
    tasks.append(create_task(
        name="count_vowels_and_consonants",
        signature="def count_vowels_and_consonants(s: str) -> dict[str, int]:",
        doc_desc="Counts alphabetic vowels and consonants in string s, returning {'vowels': V, 'consonants': C}.",
        doctests=[">>> count_vowels_and_consonants('Hello, World!')", "{'vowels': 3, 'consonants': 7}"],
        hint="Convert to lowercase. Loop chars: if char.isalpha(), check if in 'aeiou' for vowels; otherwise consonants.",
        solution="    vowels = set('aeiou')\n    v, c = 0, 0\n    for ch in s.lower():\n        if ch.isalpha():\n            if ch in vowels:\n                v += 1\n            else:\n                c += 1\n    return {'vowels': v, 'consonants': c}",
        test="assert count_vowels_and_consonants('Hello, World!') == {'vowels': 3, 'consonants': 7}\nassert count_vowels_and_consonants('1234!') == {'vowels': 0, 'consonants': 0}\nassert count_vowels_and_consonants('') == {'vowels': 0, 'consonants': 0}\n"
    ))

    # 12. Is Pangram Sentence
    tasks.append(create_task(
        name="is_pangram_sentence",
        signature="def is_pangram_sentence(sentence: str) -> bool:",
        doc_desc="Checks if a sentence contains every letter of the English alphabet at least once (case-insensitive).",
        doctests=[">>> is_pangram_sentence('The quick brown fox jumps over the lazy dog')", "True"],
        hint="Collect lowercase alphabetic letters into a set: len({ch for ch in sentence.lower() if 'a' <= ch <= 'z'}) == 26.",
        solution="    letters = {ch for ch in sentence.lower() if 'a' <= ch <= 'z'}\n    return len(letters) == 26",
        test="assert is_pangram_sentence('The quick brown fox jumps over the lazy dog') is True\nassert is_pangram_sentence('Hello World') is False\nassert is_pangram_sentence('') is False\n"
    ))

    # 13. Is Palindrome Sentence
    tasks.append(create_task(
        name="is_palindrome_sentence",
        signature="def is_palindrome_sentence(s: str) -> bool:",
        doc_desc="Checks if sentence is a palindrome, considering only alphanumeric characters and ignoring case.",
        doctests=[">>> is_palindrome_sentence('A man, a plan, a canal: Panama')", "True"],
        hint="Filter alphanumeric characters: filtered = [ch.lower() for ch in s if ch.isalnum()]. Check filtered == filtered[::-1].",
        solution="    filtered = [ch.lower() for ch in s if ch.isalnum()]\n    return filtered == filtered[::-1]",
        test="assert is_palindrome_sentence('A man, a plan, a canal: Panama') is True\nassert is_palindrome_sentence('race a car') is False\nassert is_palindrome_sentence(' ') is True\n"
    ))

    # 14. Count Substring Non-Overlapping
    tasks.append(create_task(
        name="count_substring_non_overlapping",
        signature="def count_substring_non_overlapping(text: str, sub: str) -> int:",
        doc_desc="Counts non-overlapping occurrences of substring sub in text. Returns 0 if sub is empty.",
        doctests=[">>> count_substring_non_overlapping('aaaa', 'aa')", "2"],
        hint="Use str.count(sub) if sub is not empty, otherwise return 0.",
        solution="    if not sub:\n        return 0\n    return text.count(sub)",
        test="assert count_substring_non_overlapping('aaaa', 'aa') == 2\nassert count_substring_non_overlapping('banana', 'an') == 2\nassert count_substring_non_overlapping('hello', '') == 0\nassert count_substring_non_overlapping('', 'a') == 0\n"
    ))

    # 15. Longest Palindromic Substring
    tasks.append(create_task(
        name="longest_palindromic_substring",
        signature="def longest_palindromic_substring(s: str) -> str:",
        doc_desc="Finds the longest contiguous palindromic substring in string s.",
        doctests=[">>> longest_palindromic_substring('babad') in ('bab', 'aba')", "True"],
        hint="Expand around centers for each index i (both odd center (i, i) and even center (i, i+1)). Track max length palindrome.",
        solution="    if not s:\n        return ''\n    start, end = 0, 0\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1\n            r += 1\n        return l + 1, r - 1\n    for i in range(len(s)):\n        l1, r1 = expand(i, i)\n        l2, r2 = expand(i, i + 1)\n        if r1 - l1 > end - start:\n            start, end = l1, r1\n        if r2 - l2 > end - start:\n            start, end = l2, r2\n    return s[start:end+1]",
        test="assert longest_palindromic_substring('babad') in ('bab', 'aba')\nassert longest_palindromic_substring('cbbd') == 'bb'\nassert longest_palindromic_substring('a') == 'a'\nassert longest_palindromic_substring('') == ''\n"
    ))

    # 16. Levenshtein Distance
    tasks.append(create_task(
        name="levenshtein_distance",
        signature="def levenshtein_distance(s1: str, s2: str) -> int:",
        doc_desc="Computes minimum edit distance (insertions, deletions, substitutions) between two strings.",
        doctests=[">>> levenshtein_distance('kitten', 'sitting')", "3"],
        hint="DP table of size (len(s1)+1) x (len(s2)+1): dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (0 if s1[i-1]==s2[j-1] else 1)).",
        solution="    m, n = len(s1), len(s2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(m + 1): dp[i][0] = i\n    for j in range(n + 1): dp[0][j] = j\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            cost = 0 if s1[i - 1] == s2[j - 1] else 1\n            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)\n    return dp[m][n]",
        test="assert levenshtein_distance('kitten', 'sitting') == 3\nassert levenshtein_distance('flaw', 'lawn') == 2\nassert levenshtein_distance('', '') == 0\nassert levenshtein_distance('abc', '') == 3\n"
    ))

    # 17. Strip HTML Tags Simple
    tasks.append(create_task(
        name="strip_html_tags_simple",
        signature="def strip_html_tags_simple(html: str) -> str:",
        doc_desc="Removes HTML tags from string using regex, returning clean inner text.",
        doctests=[">>> strip_html_tags_simple('<p>Hello <b>World</b>!</p>')", "'Hello World!'"],
        hint="Use regex re.sub(r'<[^>]+>', '', html).",
        solution="    import re\n    return re.sub(r'<[^>]+>', '', html)",
        test="assert strip_html_tags_simple('<p>Hello <b>World</b>!</p>') == 'Hello World!'\nassert strip_html_tags_simple('No tags here.') == 'No tags here.'\nassert strip_html_tags_simple('<div class=\"main\"><span/></div>') == ''\n"
    ))

    # 18. Extract Hashtags Text
    tasks.append(create_task(
        name="extract_hashtags_text",
        signature="def extract_hashtags_text(text: str) -> list[str]:",
        doc_desc="Extracts all hashtag words (starting with # and followed by alphanumeric characters/underscores).",
        doctests=[">>> extract_hashtags_text('Trending on #twitter: #ai_tech and #ML!')", "['#twitter', '#ai_tech', '#ML']"],
        hint="Use regex re.findall(r'#\\w+', text).",
        solution="    import re\n    return re.findall(r'#\\w+', text)",
        test="assert extract_hashtags_text('Trending on #twitter: #ai_tech and #ML!') == ['#twitter', '#ai_tech', '#ML']\nassert extract_hashtags_text('No hashtags here') == []\nassert extract_hashtags_text('#single') == ['#single']\n"
    ))

    # 19. Parse Key Value Query String
    tasks.append(create_task(
        name="parse_query_string",
        signature="def parse_query_string(query: str) -> dict[str, str]:",
        doc_desc="Parses URL query string like 'a=1&b=two' into a dictionary of key-value pairs.",
        doctests=[">>> parse_query_string('name=alice&age=30')", "{'name': 'alice', 'age': '30'}"],
        hint="Strip leading '?'. Split by '&'. For each pair, split by '=' up to 1 split.",
        solution="    q = query.lstrip('?')\n    if not q:\n        return {}\n    res = {}\n    for pair in q.split('&'):\n        if '=' in pair:\n            k, v = pair.split('=', 1)\n            res[k] = v\n    return res",
        test="assert parse_query_string('name=alice&age=30') == {'name': 'alice', 'age': '30'}\nassert parse_query_string('?page=2&limit=50') == {'page': '2', 'limit': '50'}\nassert parse_query_string('') == {}\n"
    ))

    # 20. Format Phone Number Standard
    tasks.append(create_task(
        name="format_phone_number_standard",
        signature="def format_phone_number_standard(digits_str: str) -> str | None:",
        doc_desc="Extracts digits from input and formats as '(XXX) XXX-XXXX' if exactly 10 digits are found, else None.",
        doctests=[">>> format_phone_number_standard('1234567890')", "'(123) 456-7890'"],
        hint="Filter digits: digits = [ch for ch in digits_str if ch.isdigit()]. If len != 10 return None; else format f'({d[:3]}) {d[3:6]}-{d[6:]}'.",
        solution="    digits = [ch for ch in digits_str if ch.isdigit()]\n    if len(digits) != 10:\n        return None\n    d = ''.join(digits)\n    return f'({d[:3]}) {d[3:6]}-{d[6:]}'",
        test="assert format_phone_number_standard('1234567890') == '(123) 456-7890'\nassert format_phone_number_standard('123-456-7890') == '(123) 456-7890'\nassert format_phone_number_standard('123') is None\n"
    ))

    # 21. Wrap Text to Width
    tasks.append(create_task(
        name="wrap_text_to_width",
        signature="def wrap_text_to_width(text: str, width: int) -> list[str]:",
        doc_desc="Wraps text into lines of at most width characters without breaking words.",
        doctests=[">>> wrap_text_to_width('The quick brown fox', 10)", "['The quick', 'brown fox']"],
        hint="Use textwrap.wrap(text, width=width) or greedy word accumulation checking line length.",
        solution="    import textwrap\n    return textwrap.wrap(text, width=width)",
        test="assert wrap_text_to_width('The quick brown fox', 10) == ['The quick', 'brown fox']\nassert wrap_text_to_width('Short', 10) == ['Short']\nassert wrap_text_to_width('', 10) == []\n"
    ))

    # 22. Truncate With Ellipsis
    tasks.append(create_task(
        name="truncate_with_ellipsis",
        signature="def truncate_with_ellipsis(text: str, max_length: int) -> str:",
        doc_desc="Truncates text to max_length adding '...' at the end if truncated. If max_length <= 3 and truncated, return '...'[:max_length].",
        doctests=[">>> truncate_with_ellipsis('Hello World', 8)", "'Hello...'"],
        hint="If len(text) <= max_length return text. If max_length <= 3 return '...'[:max_length]. Otherwise return text[:max_length - 3] + '...'.",
        solution="    if len(text) <= max_length:\n        return text\n    if max_length <= 3:\n        return '...'[:max_length]\n    return text[:max_length - 3] + '...'",
        test="assert truncate_with_ellipsis('Hello World', 8) == 'Hello...'\nassert truncate_with_ellipsis('Hi', 5) == 'Hi'\nassert truncate_with_ellipsis('Hello World', 2) == '..'\n"
    ))

    # 23. Expand Tabs to Spaces
    tasks.append(create_task(
        name="expand_tabs_to_spaces",
        signature="def expand_tabs_to_spaces(text: str, tab_size: int = 4) -> str:",
        doc_desc="Replaces each tab character with spaces aligning to the tab stops of size tab_size.",
        doctests=[">>> expand_tabs_to_spaces('a\\tb', 4)", "'a   b'"],
        hint="Use str.expandtabs(tab_size).",
        solution="    return text.expandtabs(tab_size)",
        test="assert expand_tabs_to_spaces('a\\tb', 4) == 'a   b'\nassert expand_tabs_to_spaces('abcd\\te', 4) == 'abcd    e'\nassert expand_tabs_to_spaces('no tabs') == 'no tabs'\n"
    ))

    # 24. Unescape Basic HTML Entities
    tasks.append(create_task(
        name="unescape_html_entities",
        signature="def unescape_html_entities(text: str) -> str:",
        doc_desc="Replaces common HTML entities (&amp;, &lt;, &gt;, &quot;, &#39;) with their literal characters.",
        doctests=[">>> unescape_html_entities('Tom &amp; Jerry &lt;3')", "'Tom & Jerry <3'"],
        hint="Use html.unescape(text).",
        solution="    import html\n    return html.unescape(text)",
        test="assert unescape_html_entities('Tom &amp; Jerry &lt;3') == 'Tom & Jerry <3'\nassert unescape_html_entities('&quot;Quote&#39;') == '\"Quote\\''\nassert unescape_html_entities('Normal text') == 'Normal text'\n"
    ))

    # 25. Compress String Consecutive
    tasks.append(create_task(
        name="compress_string_consecutive",
        signature="def compress_string_consecutive(s: str) -> str:",
        doc_desc="Compresses consecutive repeating characters to char + count only if compressed length is strictly less than original.",
        doctests=[">>> compress_string_consecutive('aabcccccaaa')", "'a2b1c5a3'"],
        hint="Run-length encode with groupby. If len(compressed) < len(s), return compressed; else return s.",
        solution="    from itertools import groupby\n    if not s: return ''\n    comp = ''.join(f'{k}{len(list(g))}' for k, g in groupby(s))\n    return comp if len(comp) < len(s) else s",
        test="assert compress_string_consecutive('aabcccccaaa') == 'a2b1c5a3'\nassert compress_string_consecutive('abcdef') == 'abcdef'\nassert compress_string_consecutive('') == ''\n"
    ))

    # 26. Remove Duplicate Words Preserve Order
    tasks.append(create_task(
        name="remove_duplicate_words_preserve_order",
        signature="def remove_duplicate_words_preserve_order(s: str) -> str:",
        doc_desc="Removes consecutive and non-consecutive duplicate words from a string, preserving first occurrence order.",
        doctests=[">>> remove_duplicate_words_preserve_order('alpha beta beta gamma alpha')", "'alpha beta gamma'"],
        hint="Split by whitespace. Use seen set and list to append words seen for the first time; join with space.",
        solution="    words = s.split()\n    seen = set()\n    res = []\n    for w in words:\n        if w not in seen:\n            seen.add(w)\n            res.append(w)\n    return ' '.join(res)",
        test="assert remove_duplicate_words_preserve_order('alpha beta beta gamma alpha') == 'alpha beta gamma'\nassert remove_duplicate_words_preserve_order('one two three') == 'one two three'\nassert remove_duplicate_words_preserve_order('') == ''\n"
    ))

    # 27. Parse CSV Line Quoted
    tasks.append(create_task(
        name="parse_csv_line_quoted",
        signature="def parse_csv_line_quoted(line: str) -> list[str]:",
        doc_desc="Parses a single CSV line with support for commas inside double-quoted fields.",
        doctests=[">>> parse_csv_line_quoted('1,\"hello, world\",3')", "['1', 'hello, world', '3']"],
        hint="Use Python's built-in csv.reader with io.StringIO(line).",
        solution="    import csv\n    import io\n    reader = csv.reader(io.StringIO(line))\n    for row in reader:\n        return row\n    return []",
        test="assert parse_csv_line_quoted('1,\"hello, world\",3') == ['1', 'hello, world', '3']\nassert parse_csv_line_quoted('a,b,c') == ['a', 'b', 'c']\nassert parse_csv_line_quoted('') == []\n"
    ))

    # 28. Format Currency USD
    tasks.append(create_task(
        name="format_currency_usd",
        signature="def format_currency_usd(amount: float) -> str:",
        doc_desc="Formats a floating-point number into USD currency string with commas and 2 decimals (e.g. 1234.5 -> '$1,234.50'). Supports negative numbers as '-$X'.",
        doctests=[">>> format_currency_usd(1234.5)", "'$1,234.50'"],
        hint="Handle negative numbers: if amount < 0, format abs(amount) and prefix with '-$'. For non-negative use f'${amount:,.2f}'.",
        solution="    if amount < 0:\n        return f'-${abs(amount):,.2f}'\n    return f'${amount:,.2f}'",
        test="assert format_currency_usd(1234.5) == '$1,234.50'\nassert format_currency_usd(-50.0) == '-$50.00'\nassert format_currency_usd(0.0) == '$0.00'\n"
    ))

    # 29. Slugify String
    tasks.append(create_task(
        name="slugify_string",
        signature="def slugify_string(title: str) -> str:",
        doc_desc="Converts title to URL-friendly lowercase slug replacing non-alphanumerics with hyphens and stripping trailing hyphens.",
        doctests=[">>> slugify_string('Hello, World! 2026')", "'hello-world-2026'"],
        hint="Use regex re.sub(r'[^a-zA-Z0-9]+', '-', title).strip('-').lower().",
        solution="    import re\n    clean = re.sub(r'[^a-zA-Z0-9]+', '-', title)\n    return clean.strip('-').lower()",
        test="assert slugify_string('Hello, World! 2026') == 'hello-world-2026'\nassert slugify_string('---A--B---') == 'a-b'\nassert slugify_string('') == ''\n"
    ))

    # 30. Reverse Words in Sentence
    tasks.append(create_task(
        name="reverse_words_in_sentence",
        signature="def reverse_words_in_sentence(s: str) -> str:",
        doc_desc="Reverses the order of words in string s, collapsing multiple whitespaces into single space.",
        doctests=[">>> reverse_words_in_sentence('  the sky   is blue  ')", "'blue is sky the'"],
        hint="Split by whitespace s.split(), reverse the list, and join with space.",
        solution="    return ' '.join(reversed(s.split()))",
        test="assert reverse_words_in_sentence('  the sky   is blue  ') == 'blue is sky the'\nassert reverse_words_in_sentence('hello world') == 'world hello'\nassert reverse_words_in_sentence('') == ''\n"
    ))

    # 31. Mask Sensitive Digits
    tasks.append(create_task(
        name="mask_sensitive_digits",
        signature="def mask_sensitive_digits(card_number: str) -> str:",
        doc_desc="Masks all digits in card number string with '*' except the last 4 digits.",
        doctests=[">>> mask_sensitive_digits('1234-5678-9012-3456')", "'****-****-****-3456'"],
        hint="Identify the indices of all digits in card_number. Leave the last 4 unchanged; replace earlier digits with '*'.",
        solution="    digits_idx = [i for i, ch in enumerate(card_number) if ch.isdigit()]\n    mask_indices = set(digits_idx[:-4]) if len(digits_idx) > 4 else set()\n    return ''.join('*' if i in mask_indices else ch for i, ch in enumerate(card_number))",
        test="assert mask_sensitive_digits('1234-5678-9012-3456') == '****-****-****-3456'\nassert mask_sensitive_digits('1234') == '1234'\nassert mask_sensitive_digits('12345') == '*2345'\n"
    ))

    # 32. Decode String Bracket Multipliers
    tasks.append(create_task(
        name="decode_string_bracket_multipliers",
        signature="def decode_string_bracket_multipliers(s: str) -> str:",
        doc_desc="Decodes string with nested pattern k[encoded_string] (e.g. '3[a]2[bc]' -> 'aaabcbc').",
        doctests=[">>> decode_string_bracket_multipliers('3[a]2[bc]')", "'aaabcbc'"],
        hint="Use a stack storing (curr_str, curr_num). When '[' is hit, push and reset. When ']' is hit, pop prev and multiply.",
        solution="    stack = []\n    curr_str = ''\n    curr_num = 0\n    for ch in s:\n        if ch.isdigit():\n            curr_num = curr_num * 10 + int(ch)\n        elif ch == '[':\n            stack.append((curr_str, curr_num))\n            curr_str = ''\n            curr_num = 0\n        elif ch == ']':\n            prev_str, num = stack.pop()\n            curr_str = prev_str + curr_str * num\n        else:\n            curr_str += ch\n    return curr_str",
        test="assert decode_string_bracket_multipliers('3[a]2[bc]') == 'aaabcbc'\nassert decode_string_bracket_multipliers('3[a2[c]]') == 'accaccacc'\nassert decode_string_bracket_multipliers('2[abc]3[cd]ef') == 'abcabccdcdcdef'\n"
    ))

    # 33. Count and Say Sequence
    tasks.append(create_task(
        name="count_and_say_sequence",
        signature="def count_and_say_sequence(n: int) -> str:",
        doc_desc="Returns the nth term of the count-and-say sequence where term 1 is '1'.",
        doctests=[">>> count_and_say_sequence(4)", "'1211'"],
        hint="Start with '1'. For each step up to n, use itertools.groupby to construct ''.join(f'{len(list(g))}{k}' for k, g in groupby(curr)).",
        solution="    from itertools import groupby\n    curr = '1'\n    for _ in range(n - 1):\n        curr = ''.join(f'{len(list(g))}{k}' for k, g in groupby(curr))\n    return curr",
        test="assert count_and_say_sequence(1) == '1'\nassert count_and_say_sequence(4) == '1211'\nassert count_and_say_sequence(5) == '111221'\n"
    ))

    # 34. Valid IPv4 Address Check
    tasks.append(create_task(
        name="is_valid_ipv4",
        signature="def is_valid_ipv4(ip: str) -> bool:",
        doc_desc="Validates whether string is a valid IPv4 address (4 octets 0..255 with no leading zeros).",
        doctests=[">>> is_valid_ipv4('192.168.1.1')", "True", ">>> is_valid_ipv4('256.1.1.1')", "False"],
        hint="Split by '.'. Check exactly 4 parts; each part must be digits only, between 0 and 255, and no leading zero if len > 1.",
        solution="    parts = ip.split('.')\n    if len(parts) != 4:\n        return False\n    for p in parts:\n        if not p.isdigit() or (len(p) > 1 and p[0] == '0'):\n            return False\n        if not (0 <= int(p) <= 255):\n            return False\n    return True",
        test="assert is_valid_ipv4('192.168.1.1') is True\nassert is_valid_ipv4('256.1.1.1') is False\nassert is_valid_ipv4('192.168.01.1') is False\nassert is_valid_ipv4('192.168.1') is False\n"
    ))

    # 35. Simplify Unix File Path
    tasks.append(create_task(
        name="simplify_unix_path",
        signature="def simplify_unix_path(path: str) -> str:",
        doc_desc="Simplifies an absolute Unix-style file path resolving '.' and '..' components.",
        doctests=[">>> simplify_unix_path('/a/./b/../../c/')", "'/c'"],
        hint="Split by '/'. Use stack: if part == '..': pop if stack; elif part not in ('', '.'): push. Return '/' + '/'.join(stack).",
        solution="    parts = path.split('/')\n    stack = []\n    for p in parts:\n        if p == '..':\n            if stack: stack.pop()\n        elif p and p != '.':\n            stack.append(p)\n    return '/' + '/'.join(stack)",
        test="assert simplify_unix_path('/a/./b/../../c/') == '/c'\nassert simplify_unix_path('/home//foo/') == '/home/foo'\nassert simplify_unix_path('/../') == '/'\nassert simplify_unix_path('/home/user/Documents/../Pictures') == '/home/user/Pictures'\n"
    ))

    # 36. Remove Adjacent Duplicate Letters
    tasks.append(create_task(
        name="remove_adjacent_duplicates",
        signature="def remove_adjacent_duplicates(s: str) -> str:",
        doc_desc="Repeatedly removes two adjacent equal letters until no adjacent duplicates remain.",
        doctests=[">>> remove_adjacent_duplicates('abbaca')", "'ca'"],
        hint="Use a stack: for each char, if stack and stack[-1] == ch: stack.pop(); else: stack.append(ch). Return ''.join(stack).",
        solution="    stack = []\n    for ch in s:\n        if stack and stack[-1] == ch:\n            stack.pop()\n        else:\n            stack.append(ch)\n    return ''.join(stack)",
        test="assert remove_adjacent_duplicates('abbaca') == 'ca'\nassert remove_adjacent_duplicates('azxxzy') == 'ay'\nassert remove_adjacent_duplicates('') == ''\n"
    ))

    # 37. Add Binary Strings
    tasks.append(create_task(
        name="add_binary_strings",
        signature="def add_binary_strings(a: str, b: str) -> str:",
        doc_desc="Calculates the sum of two binary strings returning the sum as a binary string.",
        doctests=[">>> add_binary_strings('11', '1')", "'100'"],
        hint="Convert using bin(int(a, 2) + int(b, 2))[2:].",
        solution="    return bin(int(a, 2) + int(b, 2))[2:]",
        test="assert add_binary_strings('11', '1') == '100'\nassert add_binary_strings('1010', '1011') == '10101'\nassert add_binary_strings('0', '0') == '0'\n"
    ))

    # 38. Compare Version Numbers
    tasks.append(create_task(
        name="compare_version_numbers",
        signature="def compare_version_numbers(v1: str, v2: str) -> int:",
        doc_desc="Compares two version strings. Returns 1 if v1 > v2, -1 if v1 < v2, and 0 if equal (ignoring trailing zeros).",
        doctests=[">>> compare_version_numbers('1.01', '1.001')", "0", ">>> compare_version_numbers('1.0', '1.0.0')", "0"],
        hint="Split by '.', cast to ints, zip_longest with fillvalue=0, and compare component by component.",
        solution="    from itertools import zip_longest\n    nums1 = [int(x) for x in v1.split('.')]\n    nums2 = [int(x) for x in v2.split('.')]\n    for p1, p2 in zip_longest(nums1, nums2, fillvalue=0):\n        if p1 > p2: return 1\n        elif p1 < p2: return -1\n    return 0",
        test="assert compare_version_numbers('1.01', '1.001') == 0\nassert compare_version_numbers('1.0', '1.0.0') == 0\nassert compare_version_numbers('0.1', '1.1') == -1\nassert compare_version_numbers('1.2', '1.10') == -1\nassert compare_version_numbers('2.0.1', '2.0') == 1\n"
    ))

    # 39. Longest Substring Without Repeating Characters
    tasks.append(create_task(
        name="length_of_longest_unique_substring",
        signature="def length_of_longest_unique_substring(s: str) -> int:",
        doc_desc="Finds length of longest substring without repeating characters in O(n) time.",
        doctests=[">>> length_of_longest_unique_substring('abcabcbb')", "3"],
        hint="Sliding window with dict storing last seen index: left = max(left, last_seen[ch] + 1).",
        solution="    last_seen = {}\n    left = 0\n    max_len = 0\n    for right, ch in enumerate(s):\n        if ch in last_seen and last_seen[ch] >= left:\n            left = last_seen[ch] + 1\n        last_seen[ch] = right\n        max_len = max(max_len, right - left + 1)\n    return max_len",
        test="assert length_of_longest_unique_substring('abcabcbb') == 3\nassert length_of_longest_unique_substring('bbbbb') == 1\nassert length_of_longest_unique_substring('pwwkew') == 3\nassert length_of_longest_unique_substring('') == 0\n"
    ))

    # 40. Custom Sort String by Order
    tasks.append(create_task(
        name="custom_sort_string",
        signature="def custom_sort_string(order: str, s: str) -> str:",
        doc_desc="Permutes string s so its characters appear in the custom relative order defined by order string.",
        doctests=[">>> custom_sort_string('cba', 'abcd')", "'cbad'"],
        hint="Map chars in order to index weights; chars not in order get index len(order). Sort s by this key.",
        solution="    rank = {ch: i for i, ch in enumerate(order)}\n    return ''.join(sorted(s, key=lambda ch: rank.get(ch, len(order))))",
        test="assert custom_sort_string('cba', 'abcd') in ('cbad', 'cbda')\nassert custom_sort_string('cbafg', 'abcd') in ('cbad', 'cbda')\nassert custom_sort_string('abc', '') == ''\n"
    ))

    # 41. Count Palindromic Substrings
    tasks.append(create_task(
        name="count_palindromic_substrings",
        signature="def count_palindromic_substrings(s: str) -> int:",
        doc_desc="Counts the total number of palindromic substrings in string s.",
        doctests=[">>> count_palindromic_substrings('abc')", "3", ">>> count_palindromic_substrings('aaa')", "6"],
        hint="Expand around centers for both (i, i) and (i, i+1), incrementing count while characters match.",
        solution="    count = 0\n    def expand(l, r):\n        cnt = 0\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            cnt += 1\n            l -= 1\n            r += 1\n        return cnt\n    for i in range(len(s)):\n        count += expand(i, i) + expand(i, i + 1)\n    return count",
        test="assert count_palindromic_substrings('abc') == 3\nassert count_palindromic_substrings('aaa') == 6\nassert count_palindromic_substrings('') == 0\n"
    ))

    # 42. Partition Labels Last Occurrence
    tasks.append(create_task(
        name="partition_labels",
        signature="def partition_labels(s: str) -> list[int]:",
        doc_desc="Partitions string s into as many parts as possible such that each letter appears in at most one part.",
        doctests=[">>> partition_labels('ababcbacadefegdehijhklij')", "[9, 7, 8]"],
        hint="Record last seen index of each char. Track end = max(end, last[ch]). When current index reaches end, record partition length.",
        solution="    last = {ch: i for i, ch in enumerate(s)}\n    res = []\n    start, end = 0, 0\n    for i, ch in enumerate(s):\n        end = max(end, last[ch])\n        if i == end:\n            res.append(end - start + 1)\n            start = i + 1\n    return res",
        test="assert partition_labels('ababcbacadefegdehijhklij') == [9, 7, 8]\nassert partition_labels('eccbbbbdec') == [10]\nassert partition_labels('') == []\n"
    ))

    # 43. String to Integer Atoi
    tasks.append(create_task(
        name="string_to_integer_atoi",
        signature="def string_to_integer_atoi(s: str) -> int:",
        doc_desc="Parses leading signed integer from string, clamping to 32-bit signed range [-2^31, 2^31 - 1].",
        doctests=[">>> string_to_integer_atoi('42')", "42", ">>> string_to_integer_atoi('   -42')", "-42"],
        hint="Strip leading whitespace. Detect optional '+' or '-'. Accumulate digits until non-digit. Clamp to [-2**31, 2**31 - 1].",
        solution="    s = s.lstrip()\n    if not s:\n        return 0\n    sign = 1\n    idx = 0\n    if s[0] in ('+', '-'):\n        sign = -1 if s[0] == '-' else 1\n        idx = 1\n    num = 0\n    while idx < len(s) and s[idx].isdigit():\n        num = num * 10 + int(s[idx])\n        idx += 1\n    res = sign * num\n    INT_MIN, INT_MAX = -2**31, 2**31 - 1\n    return max(INT_MIN, min(INT_MAX, res))",
        test="assert string_to_integer_atoi('42') == 42\nassert string_to_integer_atoi('   -42') == -42\nassert string_to_integer_atoi('4193 with words') == 4193\nassert string_to_integer_atoi('-91283472332') == -2**31\n"
    ))

    # 44. Find Common Characters
    tasks.append(create_task(
        name="find_common_characters",
        signature="def find_common_characters(words: list[str]) -> list[str]:",
        doc_desc="Returns a list of characters that appear in every word, including duplicates.",
        doctests=[">>> sorted(find_common_characters(['bella', 'label', 'roller']))", "['e', 'l', 'l']"],
        hint="Use collections.Counter for each word and take intersection (minimum count across counters).",
        solution="    from collections import Counter\n    if not words:\n        return []\n    common = Counter(words[0])\n    for w in words[1:]:\n        common &= Counter(w)\n    return list(common.elements())",
        test="assert sorted(find_common_characters(['bella', 'label', 'roller'])) == ['e', 'l', 'l']\nassert sorted(find_common_characters(['cool', 'lock', 'cook'])) == ['c', 'o']\nassert find_common_characters([]) == []\n"
    ))

    # 45. Capitalize Title Custom
    tasks.append(create_task(
        name="capitalize_title_custom",
        signature="def capitalize_title_custom(title: str, stop_words: list[str] | None = None) -> str:",
        doc_desc="Capitalizes each word in a title except stop words (unless the stop word is the first word).",
        doctests=[">>> capitalize_title_custom('the lord of the rings', ['the', 'of'])", "'The Lord of the Rings'"],
        hint="Lowercase stop_words. First word is always capitalized; subsequent words are capitalized only if not in stop_words.",
        solution="    stops = set(w.lower() for w in (stop_words or []))\n    words = title.split()\n    if not words:\n        return ''\n    res = [words[0].capitalize()]\n    for w in words[1:]:\n        res.append(w.lower() if w.lower() in stops else w.capitalize())\n    return ' '.join(res)",
        test="assert capitalize_title_custom('the lord of the rings', ['the', 'of']) == 'The Lord of the Rings'\nassert capitalize_title_custom('a story of hope', ['of', 'a']) == 'A Story of Hope'\nassert capitalize_title_custom('') == ''\n"
    ))

    # 46. Count Sentences in Text
    tasks.append(create_task(
        name="count_sentences_in_text",
        signature="def count_sentences_in_text(text: str) -> int:",
        doc_desc="Counts number of sentences in text based on terminating punctuation ('.', '!', '?').",
        doctests=[">>> count_sentences_in_text('Hello world! How are you? Fine.')", "3"],
        hint="Use regex re.split(r'[.!?]+', text) and count non-empty stripped segments.",
        solution="    import re\n    segments = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]\n    return len(segments)",
        test="assert count_sentences_in_text('Hello world! How are you? Fine.') == 3\nassert count_sentences_in_text('Single sentence.') == 1\nassert count_sentences_in_text('') == 0\n"
    ))

    # 47. Remove Python Line Comments
    tasks.append(create_task(
        name="remove_python_line_comments",
        signature="def remove_python_line_comments(code: str) -> str:",
        doc_desc="Removes trailing '#' comments from each line of code, stripping trailing spaces.",
        doctests=[">>> remove_python_line_comments('x = 1 # assign x')", "'x = 1'"],
        hint="Split lines. For each line, take line.split('#')[0].rstrip() and rejoin with newline.",
        solution="    lines = code.split('\\\\n')\n    return '\\\\n'.join(line.split('#')[0].rstrip() for line in lines)",
        test="assert remove_python_line_comments('x = 1 # assign x') == 'x = 1'\nassert remove_python_line_comments('y = 2 # test') == 'y = 2'\nassert remove_python_line_comments('a = 5') == 'a = 5'\n"
    ))

    # 48. Extract Email Domains
    tasks.append(create_task(
        name="extract_email_domains",
        signature="def extract_email_domains(emails: list[str]) -> list[str]:",
        doc_desc="Extracts unique domain names from a list of email addresses, sorted alphabetically.",
        doctests=[">>> extract_email_domains(['alice@gmail.com', 'bob@yahoo.com', 'charlie@gmail.com'])", "['gmail.com', 'yahoo.com']"],
        hint="Extract part after '@' for each email containing '@'. Return sorted unique list.",
        solution="    domains = {e.split('@')[1].lower() for e in emails if '@' in e and not e.endswith('@')}\n    return sorted(list(domains))",
        test="assert extract_email_domains(['alice@gmail.com', 'bob@yahoo.com', 'charlie@gmail.com']) == ['gmail.com', 'yahoo.com']\nassert extract_email_domains(['invalid']) == []\nassert extract_email_domains([]) == []\n"
    ))

    # 49. Normalize Whitespace String
    tasks.append(create_task(
        name="normalize_whitespace_string",
        signature="def normalize_whitespace_string(s: str) -> str:",
        doc_desc="Replaces consecutive whitespace characters with a single space and strips leading/trailing whitespace.",
        doctests=[">>> normalize_whitespace_string('   hello \\t  world \\n  ')", "'hello world'"],
        hint="Use ' '.join(s.split()).",
        solution="    return ' '.join(s.split())",
        test="assert normalize_whitespace_string('   hello \\t  world \\n  ') == 'hello world'\nassert normalize_whitespace_string('') == ''\nassert normalize_whitespace_string('already clean') == 'already clean'\n"
    ))

    # 50. Censor Forbidden Words
    tasks.append(create_task(
        name="censor_forbidden_words",
        signature="def censor_forbidden_words(text: str, forbidden: list[str]) -> str:",
        doc_desc="Replaces occurrences of forbidden words with asterisks of the same length, case-insensitively.",
        doctests=[">>> censor_forbidden_words('This is bad and ugly', ['bad', 'ugly'])", "'This is *** and ****'"],
        hint="Use re.sub with regex boundary r'\\b' + re.escape(word) + r'\\b' using flags=re.IGNORECASE.",
        solution="    import re\n    res = text\n    for w in forbidden:\n        pattern = re.compile(rf'\\b{re.escape(w)}\\b', re.IGNORECASE)\n        res = pattern.sub(lambda m: '*' * len(m.group(0)), res)\n    return res",
        test="assert censor_forbidden_words('This is bad and ugly', ['bad', 'ugly']) == 'This is *** and ****'\nassert censor_forbidden_words('No change', ['word']) == 'No change'\nassert censor_forbidden_words('BAD bad Bad', ['bad']) == '*** *** ***'\n"
    ))

    # 51. Alternating Caps String
    tasks.append(create_task(
        name="alternating_caps_string",
        signature="def alternating_caps_string(s: str) -> str:",
        doc_desc="Alternates case of letters (starting with uppercase for the first letter), ignoring non-letters in the sequence.",
        doctests=[">>> alternating_caps_string('hello world')", "'HeLlO wOrLd'"],
        hint="Track upper = True. For ch in s: if ch.isalpha(): ch = ch.upper() if upper else ch.lower(); upper = not upper.",
        solution="    res = []\n    upper = True\n    for ch in s:\n        if ch.isalpha():\n            res.append(ch.upper() if upper else ch.lower())\n            upper = not upper\n        else:\n            res.append(ch)\n    return ''.join(res)",
        test="assert alternating_caps_string('hello world') == 'HeLlO wOrLd'\nassert alternating_caps_string('123!') == '123!'\nassert alternating_caps_string('') == ''\n"
    ))

    # 52. Check String Rotation
    tasks.append(create_task(
        name="check_string_rotation",
        signature="def check_string_rotation(s1: str, s2: str) -> bool:",
        doc_desc="Checks if s2 is a valid rotation of s1.",
        doctests=[">>> check_string_rotation('waterbottle', 'erbottlewat')", "True", ">>> check_string_rotation('abc', 'acb')", "False"],
        hint="Check len(s1) == len(s2) and s2 in (s1 + s1).",
        solution="    return len(s1) == len(s2) and s2 in (s1 + s1)",
        test="assert check_string_rotation('waterbottle', 'erbottlewat') is True\nassert check_string_rotation('abc', 'acb') is False\nassert check_string_rotation('', '') is True\nassert check_string_rotation('a', 'a') is True\n"
    ))

    # 53. Word Frequency Counter
    tasks.append(create_task(
        name="word_frequency_map",
        signature="def word_frequency_map(text: str) -> dict[str, int]:",
        doc_desc="Returns dictionary mapping lowercase words (alphanumerics only) to their frequency.",
        doctests=[">>> word_frequency_map('Hello hello world!')", "{'hello': 2, 'world': 1}"],
        hint="Use regex re.findall(r'\\b\\w+\\b', text.lower()) and collections.Counter.",
        solution="    import re\n    from collections import Counter\n    words = re.findall(r'\\b\\w+\\b', text.lower())\n    return dict(Counter(words))",
        test="assert word_frequency_map('Hello hello world!') == {'hello': 2, 'world': 1}\nassert word_frequency_map('') == {}\nassert word_frequency_map('one') == {'one': 1}\n"
    ))

    # 54. Parse Time String to Seconds
    tasks.append(create_task(
        name="parse_time_string_to_seconds",
        signature="def parse_time_string_to_seconds(time_str: str) -> int:",
        doc_desc="Converts 'HH:MM:SS' or 'MM:SS' string into total seconds.",
        doctests=[">>> parse_time_string_to_seconds('01:02:03')", "3723", ">>> parse_time_string_to_seconds('05:30')", "330"],
        hint="Split by ':'. If 2 parts: mins, secs; if 3 parts: hours, mins, secs. Multiply accordingly.",
        solution="    parts = [int(p) for p in time_str.split(':')]\n    if len(parts) == 3:\n        return parts[0] * 3600 + parts[1] * 60 + parts[2]\n    elif len(parts) == 2:\n        return parts[0] * 60 + parts[1]\n    raise ValueError('Invalid time format')",
        test="assert parse_time_string_to_seconds('01:02:03') == 3723\nassert parse_time_string_to_seconds('05:30') == 330\nassert parse_time_string_to_seconds('00:00:00') == 0\n"
    ))

    # 55. Longest Word in Text
    tasks.append(create_task(
        name="longest_word_in_text",
        signature="def longest_word_in_text(text: str) -> str:",
        doc_desc="Returns the first longest word in text ignoring punctuation. Returns '' if no words found.",
        doctests=[">>> longest_word_in_text('The mysterious black hole appeared.')", "'mysterious'"],
        hint="Use regex re.findall(r'\\b\\w+\\b', text); return max(words, key=len, default='').",
        solution="    import re\n    words = re.findall(r'\\b\\w+\\b', text)\n    return max(words, key=len) if words else ''",
        test="assert longest_word_in_text('The mysterious black hole appeared.') == 'mysterious'\nassert longest_word_in_text('short text') == 'short'\nassert longest_word_in_text('!@#$%') == ''\n"
    ))

    # 56. Remove Specific Characters
    tasks.append(create_task(
        name="remove_specific_chars",
        signature="def remove_specific_chars(s: str, chars_to_remove: str) -> str:",
        doc_desc="Removes all occurrences of characters specified in chars_to_remove from string s.",
        doctests=[">>> remove_specific_chars('battle of the bands', 'aeiou')", "'bttl f th bnds'"],
        hint="Build remove set: bad = set(chars_to_remove). Return ''.join(ch for ch in s if ch not in bad).",
        solution="    bad = set(chars_to_remove)\n    return ''.join(ch for ch in s if ch not in bad)",
        test="assert remove_specific_chars('battle of the bands', 'aeiou') == 'bttl f th bnds'\nassert remove_specific_chars('clean', '') == 'clean'\nassert remove_specific_chars('abc', 'abc') == ''\n"
    ))

    # 57. Formatted String Table
    tasks.append(create_task(
        name="render_simple_ascii_table",
        signature="def render_simple_ascii_table(headers: list[str], rows: list[list[str]]) -> str:",
        doc_desc="Renders headers and rows into pipe-separated ASCII table with aligned columns.",
        doctests=[">>> render_simple_ascii_table(['ID', 'Name'], [['1', 'Alice'], ['2', 'Bob']]).split('\\n')[0]", "'| ID | Name  |'"],
        hint="Find max width for each column across headers and rows. Format each row with f'| {val:<width} |'.",
        solution="    cols = len(headers)\n    widths = [len(h) for h in headers]\n    for r in rows:\n        for j in range(cols):\n            if j < len(r):\n                widths[j] = max(widths[j], len(str(r[j])))\n    def fmt_row(items):\n        cells = [f'{str(items[j]):<{widths[j]}}' for j in range(cols)]\n        return '| ' + ' | '.join(cells) + ' |'\n    lines = [fmt_row(headers)]\n    sep = '|-' + '-|-'.join('-' * w for w in widths) + '-|'\n    lines.append(sep)\n    for r in rows:\n        lines.append(fmt_row(r))\n    return '\\\\n'.join(lines)",
        test="t = render_simple_ascii_table(['ID', 'Name'], [['1', 'Alice'], ['2', 'Bob']])\nassert 'Alice' in t and '|' in t\n"
    ))

    # 58. Validate Credit Card Luhn
    tasks.append(create_task(
        name="validate_luhn_algorithm",
        signature="def validate_luhn_algorithm(card_str: str) -> bool:",
        doc_desc="Validates credit card number string using Luhn check algorithm.",
        doctests=[">>> validate_luhn_algorithm('79927398713')", "True", ">>> validate_luhn_algorithm('79927398710')", "False"],
        hint="Filter digits. Starting from right, double every second digit; if doubled > 9, subtract 9. Sum all digits; return sum % 10 == 0.",
        solution="    digits = [int(d) for d in card_str if d.isdigit()]\n    if not digits:\n        return False\n    checksum = 0\n    reverse_digits = digits[::-1]\n    for i, d in enumerate(reverse_digits):\n        if i % 2 == 1:\n            d *= 2\n            if d > 9:\n                d -= 9\n        checksum += d\n    return checksum % 10 == 0",
        test="assert validate_luhn_algorithm('79927398713') is True\nassert validate_luhn_algorithm('79927398710') is False\nassert validate_luhn_algorithm('49927398716') is True\n"
    ))

    # 59. Count Words in String
    tasks.append(create_task(
        name="count_words_in_string",
        signature="def count_words_in_string(s: str) -> int:",
        doc_desc="Counts the number of whitespace-separated words in string s.",
        doctests=[">>> count_words_in_string('One two   three')", "3"],
        hint="Use len(s.split()).",
        solution="    return len(s.split())",
        test="assert count_words_in_string('One two   three') == 3\nassert count_words_in_string('') == 0\nassert count_words_in_string('   ') == 0\n"
    ))

    # 60. Check Subsequence Exists
    tasks.append(create_task(
        name="is_subsequence_string",
        signature="def is_subsequence_string(sub: str, target: str) -> bool:",
        doc_desc="Checks if sub is a subsequence of target string (characters appear in order).",
        doctests=[">>> is_subsequence_string('ace', 'abcde')", "True", ">>> is_subsequence_string('aec', 'abcde')", "False"],
        hint="Iterate target with an iterator: it = iter(target); return all(c in it for c in sub).",
        solution="    it = iter(target)\n    return all(c in it for c in sub)",
        test="assert is_subsequence_string('ace', 'abcde') is True\nassert is_subsequence_string('aec', 'abcde') is False\nassert is_subsequence_string('', 'abc') is True\n"
    ))

    # 61. Pad Left String
    tasks.append(create_task(
        name="pad_left_custom",
        signature="def pad_left_custom(s: str, length: int, pad_char: str = ' ') -> str:",
        doc_desc="Pads string on the left with pad_char until reaching length.",
        doctests=[">>> pad_left_custom('42', 5, '0')", "'00042'"],
        hint="If len(s) >= length return s. Otherwise return pad_char * (length - len(s)) + s.",
        solution="    if len(s) >= length:\n        return s\n    return pad_char * (length - len(s)) + s",
        test="assert pad_left_custom('42', 5, '0') == '00042'\nassert pad_left_custom('hello', 3) == 'hello'\nassert pad_left_custom('a', 3, '-') == '--a'\n"
    ))

    # 62. Find Substring Indices All
    tasks.append(create_task(
        name="find_all_substring_indices",
        signature="def find_all_substring_indices(text: str, sub: str) -> list[int]:",
        doc_desc="Returns all starting 0-based indices where substring sub occurs in text (supports overlapping).",
        doctests=[">>> find_all_substring_indices('aaaa', 'aa')", "[0, 1, 2]"],
        hint="Loop index from 0 to len(text) - len(sub). If text[i:i+len(sub)] == sub, record index.",
        solution="    if not sub:\n        return []\n    res = []\n    for i in range(len(text) - len(sub) + 1):\n        if text[i:i+len(sub)] == sub:\n            res.append(i)\n    return res",
        test="assert find_all_substring_indices('aaaa', 'aa') == [0, 1, 2]\nassert find_all_substring_indices('hello', 'll') == [2]\nassert find_all_substring_indices('abc', 'd') == []\nassert find_all_substring_indices('abc', '') == []\n"
    ))

    # 63. Interleave Two Strings
    tasks.append(create_task(
        name="interleave_two_strings",
        signature="def interleave_two_strings(s1: str, s2: str) -> str:",
        doc_desc="Alternates characters from s1 and s2, appending remaining characters from the longer string.",
        doctests=[">>> interleave_two_strings('abc', '12345')", "'a1b2c345'"],
        hint="Use itertools.zip_longest(s1, s2, fillvalue='') and join non-empty characters.",
        solution="    from itertools import zip_longest\n    return ''.join(a + b for a, b in zip_longest(s1, s2, fillvalue=''))",
        test="assert interleave_two_strings('abc', '12345') == 'a1b2c345'\nassert interleave_two_strings('', 'abc') == 'abc'\nassert interleave_two_strings('xy', '12') == 'x1y2'\n"
    ))

    # 64. Count Uppercase, Lowercase, Digits, Special
    tasks.append(create_task(
        name="classify_char_types_count",
        signature="def classify_char_types_count(s: str) -> dict[str, int]:",
        doc_desc="Returns counts of 'upper', 'lower', 'digits', and 'special' characters in string.",
        doctests=[">>> classify_char_types_count('Hello 123!')", "{'upper': 1, 'lower': 4, 'digits': 3, 'special': 2}"],
        hint="Iterate chars and categorize using isupper(), islower(), isdigit(), and fallback to special.",
        solution="    counts = {'upper': 0, 'lower': 0, 'digits': 0, 'special': 0}\n    for ch in s:\n        if ch.isupper(): counts['upper'] += 1\n        elif ch.islower(): counts['lower'] += 1\n        elif ch.isdigit(): counts['digits'] += 1\n        else: counts['special'] += 1\n    return counts",
        test="assert classify_char_types_count('Hello 123!') == {'upper': 1, 'lower': 4, 'digits': 3, 'special': 2}\nassert classify_char_types_count('') == {'upper': 0, 'lower': 0, 'digits': 0, 'special': 0}\n"
    ))

    return tasks
