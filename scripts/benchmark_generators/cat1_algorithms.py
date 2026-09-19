"""
Category 1: Algorithms & Numerical (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_algorithms_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Collatz sequence length
    tasks.append(create_task(
        name="collatz_sequence_length",
        signature="def collatz_sequence_length(n: int) -> int:",
        doc_desc="Returns the total number of steps to reach 1 in the Collatz 3n + 1 sequence.",
        doctests=[">>> collatz_sequence_length(1)", "0", ">>> collatz_sequence_length(6)", "8"],
        hint="Loop while n > 1. If n is even, divide by 2; otherwise compute 3 * n + 1. Increment a counter.",
        solution="    count = 0\n    while n > 1:\n        if n % 2 == 0:\n            n //= 2\n        else:\n            n = 3 * n + 1\n        count += 1\n    return count",
        test="assert collatz_sequence_length(1) == 0\nassert collatz_sequence_length(6) == 8\nassert collatz_sequence_length(27) == 111\nassert collatz_sequence_length(2) == 1\nassert collatz_sequence_length(4) == 2\n"
    ))

    # 2. Digital root
    tasks.append(create_task(
        name="digital_root",
        signature="def digital_root(n: int) -> int:",
        doc_desc="Calculates single-digit digital root by iteratively summing decimal digits.",
        doctests=[">>> digital_root(16)", "7", ">>> digital_root(942)", "6"],
        hint="While n >= 10, replace n with the sum of its digits converted from str(n).",
        solution="    while n >= 10:\n        n = sum(int(d) for d in str(n))\n    return n",
        test="assert digital_root(16) == 7\nassert digital_root(942) == 6\nassert digital_root(0) == 0\nassert digital_root(99999) == 9\nassert digital_root(10) == 1\n"
    ))

    # 3. Greatest Common Divisor
    tasks.append(create_task(
        name="greatest_common_divisor",
        signature="def greatest_common_divisor(a: int, b: int) -> int:",
        doc_desc="Computes the greatest common divisor (GCD) of two integers using the Euclidean algorithm.",
        doctests=[">>> greatest_common_divisor(48, 18)", "6", ">>> greatest_common_divisor(101, 103)", "1"],
        hint="Apply Euclidean remainder reduction: while b is non-zero, set (a, b) = (b, a % b). Return abs(a).",
        solution="    while b:\n        a, b = b, a % b\n    return abs(a)",
        test="assert greatest_common_divisor(48, 18) == 6\nassert greatest_common_divisor(101, 103) == 1\nassert greatest_common_divisor(0, 5) == 5\nassert greatest_common_divisor(-24, 16) == 8\nassert greatest_common_divisor(7, 0) == 7\n"
    ))

    # 4. Least Common Multiple
    tasks.append(create_task(
        name="least_common_multiple",
        signature="def least_common_multiple(a: int, b: int) -> int:",
        doc_desc="Computes the least common multiple (LCM) of two integers.",
        doctests=[">>> least_common_multiple(4, 6)", "12", ">>> least_common_multiple(0, 5)", "0"],
        hint="If either a or b is 0, return 0. Otherwise use the identity LCM(a, b) = abs(a * b) // GCD(a, b).",
        solution="    if a == 0 or b == 0:\n        return 0\n    x, y = abs(a), abs(b)\n    gcd_val, temp = x, y\n    while temp:\n        gcd_val, temp = temp, gcd_val % temp\n    return (x * y) // gcd_val",
        test="assert least_common_multiple(4, 6) == 12\nassert least_common_multiple(5, 7) == 35\nassert least_common_multiple(0, 10) == 0\nassert least_common_multiple(-4, 6) == 12\nassert least_common_multiple(12, 18) == 36\n"
    ))

    # 5. Armstrong Number
    tasks.append(create_task(
        name="is_armstrong_number",
        signature="def is_armstrong_number(n: int) -> bool:",
        doc_desc="Checks if non-negative integer n equals the sum of its digits each raised to the power of the number of digits.",
        doctests=[">>> is_armstrong_number(153)", "True", ">>> is_armstrong_number(10)", "False"],
        hint="For negative numbers return False. For n >= 0, find num_digits = len(str(n)), and check if sum(d**num_digits) == n.",
        solution="    if n < 0:\n        return False\n    digits = [int(d) for d in str(n)]\n    p = len(digits)\n    return sum(d ** p for d in digits) == n",
        test="assert is_armstrong_number(153) is True\nassert is_armstrong_number(370) is True\nassert is_armstrong_number(10) is False\nassert is_armstrong_number(0) is True\nassert is_armstrong_number(9) is True\nassert is_armstrong_number(9474) is True\nassert is_armstrong_number(-153) is False\n"
    ))

    # 6. Sieve of Eratosthenes
    tasks.append(create_task(
        name="sieve_of_eratosthenes",
        signature="def sieve_of_eratosthenes(limit: int) -> list[int]:",
        doc_desc="Returns a sorted list of all prime numbers up to limit inclusive.",
        doctests=[">>> sieve_of_eratosthenes(10)", "[2, 3, 5, 7]", ">>> sieve_of_eratosthenes(1)", "[]"],
        hint="Initialize a boolean list of size limit + 1. Mark 0 and 1 False. Loop up to int(limit**0.5) and mark multiples of primes False.",
        solution="    if limit < 2:\n        return []\n    is_prime = [True] * (limit + 1)\n    is_prime[0] = is_prime[1] = False\n    for i in range(2, int(limit**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i * i, limit + 1, i):\n                is_prime[j] = False\n    return [i for i, p in enumerate(is_prime) if p]",
        test="assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]\nassert sieve_of_eratosthenes(1) == []\nassert sieve_of_eratosthenes(20) == [2, 3, 5, 7, 11, 13, 17, 19]\nassert sieve_of_eratosthenes(2) == [2]\nassert sieve_of_eratosthenes(0) == []\n"
    ))

    # 7. Integer to Roman
    tasks.append(create_task(
        name="integer_to_roman",
        signature="def integer_to_roman(num: int) -> str:",
        doc_desc="Converts an integer in range 1..3999 to its standard Roman numeral string.",
        doctests=[">>> integer_to_roman(3)", "'III'", ">>> integer_to_roman(1994)", "'MCMXCIV'"],
        hint="Map values from 1000 down to 1 (including subtractives 900, 400, 90, 40, 9, 4) to their Roman symbols, subtracting greedily.",
        solution="    val_map = [\n        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),\n        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),\n        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')\n    ]\n    roman = []\n    for v, sym in val_map:\n        while num >= v:\n            roman.append(sym)\n            num -= v\n    return ''.join(roman)",
        test="assert integer_to_roman(3) == 'III'\nassert integer_to_roman(58) == 'LVIII'\nassert integer_to_roman(1994) == 'MCMXCIV'\nassert integer_to_roman(1) == 'I'\nassert integer_to_roman(3999) == 'MMMCMXCIX'\n"
    ))

    # 8. Roman to Integer
    tasks.append(create_task(
        name="roman_to_integer",
        signature="def roman_to_integer(s: str) -> int:",
        doc_desc="Converts a valid Roman numeral string into its integer value.",
        doctests=[">>> roman_to_integer('LVIII')", "58", ">>> roman_to_integer('MCMXCIV')", "1994"],
        hint="Traverse symbols from right to left. If the current value is less than the previous value, subtract it; else add it.",
        solution="    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    prev = 0\n    for ch in reversed(s):\n        curr = vals[ch]\n        if curr >= prev:\n            total += curr\n        else:\n            total -= curr\n        prev = curr\n    return total",
        test="assert roman_to_integer('III') == 3\nassert roman_to_integer('LVIII') == 58\nassert roman_to_integer('MCMXCIV') == 1994\nassert roman_to_integer('IX') == 9\nassert roman_to_integer('XL') == 40\n"
    ))

    # 9. Pascal's Triangle Row
    tasks.append(create_task(
        name="pascals_triangle_row",
        signature="def pascals_triangle_row(row_idx: int) -> list[int]:",
        doc_desc="Returns the 0-indexed row of Pascal's triangle as a list of integers.",
        doctests=[">>> pascals_triangle_row(0)", "[1]", ">>> pascals_triangle_row(3)", "[1, 3, 3, 1]"],
        hint="Start with [1]. For each row iteration, construct the new row with [1] + [row[j] + row[j+1]] + [1].",
        solution="    row = [1]\n    for _ in range(row_idx):\n        row = [1] + [row[j] + row[j+1] for j in range(len(row) - 1)] + [1]\n    return row",
        test="assert pascals_triangle_row(0) == [1]\nassert pascals_triangle_row(1) == [1, 1]\nassert pascals_triangle_row(4) == [1, 4, 6, 4, 1]\nassert pascals_triangle_row(5) == [1, 5, 10, 10, 5, 1]\n"
    ))

    # 10. Hamming Distance
    tasks.append(create_task(
        name="hamming_distance",
        signature="def hamming_distance(x: int, y: int) -> int:",
        doc_desc="Computes the number of bit positions where non-negative integers x and y differ.",
        doctests=[">>> hamming_distance(1, 4)", "2", ">>> hamming_distance(3, 1)", "1"],
        hint="Compute bitwise XOR x ^ y, convert to binary with bin(), and count the number of '1' characters.",
        solution="    return bin(x ^ y).count('1')",
        test="assert hamming_distance(1, 4) == 2\nassert hamming_distance(3, 1) == 1\nassert hamming_distance(0, 0) == 0\nassert hamming_distance(7, 0) == 3\nassert hamming_distance(15, 15) == 0\n"
    ))

    # 11. Reverse Digits Signed
    tasks.append(create_task(
        name="reverse_digits_signed",
        signature="def reverse_digits_signed(n: int) -> int:",
        doc_desc="Reverses digits of a signed integer, preserving the negative sign if present.",
        doctests=[">>> reverse_digits_signed(123)", "321", ">>> reverse_digits_signed(-450)", "-54"],
        hint="Determine sign (-1 if n < 0 else 1). Convert abs(n) to string, reverse it with [::-1], cast to int, and multiply by sign.",
        solution="    sign = -1 if n < 0 else 1\n    rev = int(str(abs(n))[::-1])\n    return sign * rev",
        test="assert reverse_digits_signed(123) == 321\nassert reverse_digits_signed(-456) == -654\nassert reverse_digits_signed(120) == 21\nassert reverse_digits_signed(0) == 0\nassert reverse_digits_signed(-5) == -5\n"
    ))

    # 12. Is Power of Two
    tasks.append(create_task(
        name="is_power_of_two",
        signature="def is_power_of_two(n: int) -> bool:",
        doc_desc="Returns True if positive integer n is a power of 2 using bitwise arithmetic.",
        doctests=[">>> is_power_of_two(16)", "True", ">>> is_power_of_two(18)", "False"],
        hint="A positive number is a power of 2 if and only if n > 0 and (n & (n - 1)) == 0.",
        solution="    return n > 0 and (n & (n - 1)) == 0",
        test="assert is_power_of_two(1) is True\nassert is_power_of_two(16) is True\nassert is_power_of_two(3) is False\nassert is_power_of_two(0) is False\nassert is_power_of_two(-8) is False\nassert is_power_of_two(1024) is True\n"
    ))

    # 13. Prime Factorization
    tasks.append(create_task(
        name="prime_factorization",
        signature="def prime_factorization(n: int) -> list[int]:",
        doc_desc="Returns the prime factors of integer n >= 2 in non-decreasing order.",
        doctests=[">>> prime_factorization(12)", "[2, 2, 3]", ">>> prime_factorization(17)", "[17]"],
        hint="Divide out factors of 2 first, then test odd numbers d starting from 3 up to sqrt(n). If n > 1 remains, append n.",
        solution="    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        factors.append(n)\n    return factors",
        test="assert prime_factorization(12) == [2, 2, 3]\nassert prime_factorization(17) == [17]\nassert prime_factorization(360) == [2, 2, 2, 3, 3, 5]\nassert prime_factorization(2) == [2]\nassert prime_factorization(49) == [7, 7]\n"
    ))

    # 14. Is Prime
    tasks.append(create_task(
        name="is_prime",
        signature="def is_prime(n: int) -> bool:",
        doc_desc="Returns True if integer n is a prime number, False otherwise.",
        doctests=[">>> is_prime(7)", "True", ">>> is_prime(4)", "False"],
        hint="Numbers <= 1 are not prime. 2 and 3 are prime. Check divisibility by 2 and 3, then test 6k +/- 1 up to sqrt(n).",
        solution="    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True",
        test="assert is_prime(7) is True\nassert is_prime(4) is False\nassert is_prime(1) is False\nassert is_prime(0) is False\nassert is_prime(-7) is False\nassert is_prime(97) is True\nassert is_prime(100) is False\n"
    ))

    # 15. Sum of Proper Divisors
    tasks.append(create_task(
        name="sum_of_proper_divisors",
        signature="def sum_of_proper_divisors(n: int) -> int:",
        doc_desc="Returns the sum of all positive divisors of n excluding n itself.",
        doctests=[">>> sum_of_proper_divisors(12)", "16", ">>> sum_of_proper_divisors(1)", "0"],
        hint="For n <= 1, return 0. Iterate d from 2 up to int(n**0.5); if d divides n, add d and (n // d if d != n // d else 0). Add 1 at the end.",
        solution="    if n <= 1:\n        return 0\n    total = 1\n    for d in range(2, int(n**0.5) + 1):\n        if n % d == 0:\n            total += d\n            if d != n // d:\n                total += n // d\n    return total",
        test="assert sum_of_proper_divisors(12) == 16\nassert sum_of_proper_divisors(6) == 6\nassert sum_of_proper_divisors(1) == 0\nassert sum_of_proper_divisors(28) == 28\nassert sum_of_proper_divisors(13) == 1\n"
    ))

    # 16. Is Perfect Number
    tasks.append(create_task(
        name="is_perfect_number",
        signature="def is_perfect_number(n: int) -> bool:",
        doc_desc="Determines whether positive integer n is a perfect number (equal to the sum of its proper positive divisors).",
        doctests=[">>> is_perfect_number(6)", "True", ">>> is_perfect_number(28)", "True", ">>> is_perfect_number(12)", "False"],
        hint="Check if n > 1 and sum_of_proper_divisors(n) == n.",
        solution="    if n <= 1:\n        return False\n    total = 1\n    for d in range(2, int(n**0.5) + 1):\n        if n % d == 0:\n            total += d\n            if d != n // d:\n                total += n // d\n    return total == n",
        test="assert is_perfect_number(6) is True\nassert is_perfect_number(28) is True\nassert is_perfect_number(12) is False\nassert is_perfect_number(1) is False\nassert is_perfect_number(496) is True\n"
    ))

    # 17. Nth Fibonacci
    tasks.append(create_task(
        name="nth_fibonacci",
        signature="def nth_fibonacci(n: int) -> int:",
        doc_desc="Computes the nth Fibonacci number (0-indexed: F(0)=0, F(1)=1).",
        doctests=[">>> nth_fibonacci(0)", "0", ">>> nth_fibonacci(7)", "13"],
        hint="Use iterative two-variable tracking (a, b = 0, 1) and advance for _ in range(n): a, b = b, a + b.",
        solution="    if n <= 0:\n        return 0\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        test="assert nth_fibonacci(0) == 0\nassert nth_fibonacci(1) == 1\nassert nth_fibonacci(7) == 13\nassert nth_fibonacci(10) == 55\nassert nth_fibonacci(12) == 144\n"
    ))

    # 18. Catalan Number
    tasks.append(create_task(
        name="catalan_number",
        signature="def catalan_number(n: int) -> int:",
        doc_desc="Calculates the nth Catalan number C(n) = (2n)! / ((n+1)! * n!).",
        doctests=[">>> catalan_number(0)", "1", ">>> catalan_number(3)", "5"],
        hint="Use math.comb(2 * n, n) // (n + 1), or compute recursively with integer multiplication.",
        solution="    import math\n    return math.comb(2 * n, n) // (n + 1)",
        test="assert catalan_number(0) == 1\nassert catalan_number(1) == 1\nassert catalan_number(2) == 2\nassert catalan_number(3) == 5\nassert catalan_number(4) == 14\nassert catalan_number(5) == 42\n"
    ))

    # 19. Modular Exponentiation
    tasks.append(create_task(
        name="modular_exponentiation",
        signature="def modular_exponentiation(base: int, exp: int, mod: int) -> int:",
        doc_desc="Computes (base ** exp) % mod efficiently using binary exponentiation.",
        doctests=[">>> modular_exponentiation(2, 10, 1000)", "24", ">>> modular_exponentiation(3, 0, 7)", "1"],
        hint="Use standard binary exponentiation: res = 1, base %= mod; while exp > 0: if exp % 2 == 1: res = (res * base) % mod; base = (base * base) % mod; exp //= 2.",
        solution="    if mod == 1:\n        return 0\n    res = 1\n    base %= mod\n    while exp > 0:\n        if exp % 2 == 1:\n            res = (res * base) % mod\n        base = (base * base) % mod\n        exp //= 2\n    return res",
        test="assert modular_exponentiation(2, 10, 1000) == 24\nassert modular_exponentiation(3, 0, 7) == 1\nassert modular_exponentiation(5, 3, 13) == 8\nassert modular_exponentiation(10, 5, 1) == 0\nassert modular_exponentiation(7, 4, 10) == 1\n"
    ))

    # 20. Extended GCD
    tasks.append(create_task(
        name="extended_gcd",
        signature="def extended_gcd(a: int, b: int) -> tuple[int, int, int]:",
        doc_desc="Computes integers (g, x, y) such that a*x + b*y = g = gcd(a, b).",
        doctests=[">>> extended_gcd(30, 20)", "(10, 1, -1)"],
        hint="Implement extended Euclidean recurrence: base case if b == 0 return (a, 1, 0); recursively call on (b, a % b) and update x, y.",
        solution="    if b == 0:\n        return (a, 1, 0)\n    g, x1, y1 = extended_gcd(b, a % b)\n    x = y1\n    y = x1 - (a // b) * y1\n    return (g, x, y)",
        test="g, x, y = extended_gcd(30, 20)\nassert 30 * x + 20 * y == g and g == 10\ng2, x2, y2 = extended_gcd(35, 15)\nassert 35 * x2 + 15 * y2 == g2 and g2 == 5\ng3, x3, y3 = extended_gcd(17, 31)\nassert 17 * x3 + 31 * y3 == g3 and g3 == 1\n"
    ))

    # 21. Binary to Gray Code
    tasks.append(create_task(
        name="binary_to_gray_code",
        signature="def binary_to_gray_code(n: int) -> int:",
        doc_desc="Converts a non-negative integer binary representation to reflected Gray code integer.",
        doctests=[">>> binary_to_gray_code(0)", "0", ">>> binary_to_gray_code(3)", "2"],
        hint="Reflected Gray code is computed simply by n ^ (n >> 1).",
        solution="    return n ^ (n >> 1)",
        test="assert binary_to_gray_code(0) == 0\nassert binary_to_gray_code(1) == 1\nassert binary_to_gray_code(2) == 3\nassert binary_to_gray_code(3) == 2\nassert binary_to_gray_code(4) == 6\nassert binary_to_gray_code(7) == 4\n"
    ))

    # 22. Gray Code to Binary
    tasks.append(create_task(
        name="gray_code_to_binary",
        signature="def gray_code_to_binary(gray: int) -> int:",
        doc_desc="Converts a reflected Gray code integer back to standard binary integer.",
        doctests=[">>> gray_code_to_binary(2)", "3", ">>> gray_code_to_binary(6)", "4"],
        hint="Initialize n = gray. While gray > 0: gray >>= 1; n ^= gray. Return n.",
        solution="    n = gray\n    while gray > 0:\n        gray >>= 1\n        n ^= gray\n    return n",
        test="assert gray_code_to_binary(0) == 0\nassert gray_code_to_binary(1) == 1\nassert gray_code_to_binary(3) == 2\nassert gray_code_to_binary(2) == 3\nassert gray_code_to_binary(6) == 4\nassert gray_code_to_binary(4) == 7\n"
    ))

    # 23. Base Conversion
    tasks.append(create_task(
        name="base_conversion",
        signature="def base_conversion(num: int, base: int) -> str:",
        doc_desc="Converts a non-negative integer to string representation in base 2..16 (uppercase letters).",
        doctests=[">>> base_conversion(10, 2)", "'1010'", ">>> base_conversion(255, 16)", "'FF'"],
        hint="If num == 0 return '0'. Build digits from num % base using '0123456789ABCDEF' and divide num //= base.",
        solution="    if num == 0:\n        return '0'\n    digits = '0123456789ABCDEF'\n    res = []\n    while num > 0:\n        res.append(digits[num % base])\n        num //= base\n    return ''.join(reversed(res))",
        test="assert base_conversion(10, 2) == '1010'\nassert base_conversion(255, 16) == 'FF'\nassert base_conversion(0, 8) == '0'\nassert base_conversion(42, 10) == '42'\nassert base_conversion(42, 8) == '52'\nassert base_conversion(11, 16) == 'B'\n"
    ))

    # 24. Is Palindrome Number
    tasks.append(create_task(
        name="is_palindrome_number",
        signature="def is_palindrome_number(n: int) -> bool:",
        doc_desc="Determines if an integer is a palindrome (reads same forward and backward). Negative numbers are not palindromes.",
        doctests=[">>> is_palindrome_number(121)", "True", ">>> is_palindrome_number(-121)", "False"],
        hint="Negative numbers return False. For n >= 0, convert to string s = str(n) and check s == s[::-1].",
        solution="    if n < 0:\n        return False\n    s = str(n)\n    return s == s[::-1]",
        test="assert is_palindrome_number(121) is True\nassert is_palindrome_number(-121) is False\nassert is_palindrome_number(10) is False\nassert is_palindrome_number(0) is True\nassert is_palindrome_number(12321) is True\n"
    ))

    # 25. Integer Square Root
    tasks.append(create_task(
        name="integer_square_root",
        signature="def integer_square_root(n: int) -> int:",
        doc_desc="Computes the floor of the square root of non-negative integer n using binary search.",
        doctests=[">>> integer_square_root(8)", "2", ">>> integer_square_root(16)", "4"],
        hint="Use binary search between 0 and n. If mid*mid <= n, update ans = mid and search right; else search left.",
        solution="    if n < 0:\n        raise ValueError('n must be non-negative')\n    low, high = 0, n\n    ans = 0\n    while low <= high:\n        mid = (low + high) // 2\n        if mid * mid <= n:\n            ans = mid\n            low = mid + 1\n        else:\n            high = mid - 1\n    return ans",
        test="assert integer_square_root(8) == 2\nassert integer_square_root(16) == 4\nassert integer_square_root(0) == 0\nassert integer_square_root(1) == 1\nassert integer_square_root(25) == 5\nassert integer_square_root(26) == 5\n"
    ))

    # 26. Trailing Zeroes in Factorial
    tasks.append(create_task(
        name="count_trailing_zeros_factorial",
        signature="def count_trailing_zeros_factorial(n: int) -> int:",
        doc_desc="Returns the count of trailing zeroes in n! (factorial) in O(log n) time.",
        doctests=[">>> count_trailing_zeros_factorial(5)", "1", ">>> count_trailing_zeros_factorial(25)", "6"],
        hint="Count the powers of 5 dividing the factors: count += n // 5, then n //= 5 while n > 0.",
        solution="    count = 0\n    while n >= 5:\n        count += n // 5\n        n //= 5\n    return count",
        test="assert count_trailing_zeros_factorial(5) == 1\nassert count_trailing_zeros_factorial(25) == 6\nassert count_trailing_zeros_factorial(0) == 0\nassert count_trailing_zeros_factorial(100) == 24\nassert count_trailing_zeros_factorial(10) == 2\n"
    ))

    # 27. Combinations Count
    tasks.append(create_task(
        name="combinations_count",
        signature="def combinations_count(n: int, k: int) -> int:",
        doc_desc="Calculates n choose k (binomial coefficient). Returns 0 if k < 0 or k > n.",
        doctests=[">>> combinations_count(5, 2)", "10", ">>> combinations_count(4, 5)", "0"],
        hint="Check boundary conditions k < 0 or k > n. Use math.comb(n, k).",
        solution="    if k < 0 or k > n:\n        return 0\n    import math\n    return math.comb(n, k)",
        test="assert combinations_count(5, 2) == 10\nassert combinations_count(4, 5) == 0\nassert combinations_count(6, 0) == 1\nassert combinations_count(6, 6) == 1\nassert combinations_count(10, 3) == 120\n"
    ))

    # 28. Permutations Count
    tasks.append(create_task(
        name="permutations_count",
        signature="def permutations_count(n: int, k: int) -> int:",
        doc_desc="Calculates the number of k-permutations of n items P(n, k) = n! / (n-k)!.",
        doctests=[">>> permutations_count(5, 2)", "20", ">>> permutations_count(3, 4)", "0"],
        hint="Return 0 if k < 0 or k > n. Otherwise use math.perm(n, k).",
        solution="    if k < 0 or k > n:\n        return 0\n    import math\n    return math.perm(n, k)",
        test="assert permutations_count(5, 2) == 20\nassert permutations_count(3, 4) == 0\nassert permutations_count(4, 0) == 1\nassert permutations_count(4, 4) == 24\nassert permutations_count(6, 3) == 120\n"
    ))

    # 29. Sum of Digits Base
    tasks.append(create_task(
        name="sum_of_digits_base",
        signature="def sum_of_digits_base(n: int, base: int) -> int:",
        doc_desc="Calculates the sum of digits of non-negative integer n when represented in the specified base.",
        doctests=[">>> sum_of_digits_base(10, 2)", "2", ">>> sum_of_digits_base(42, 10)", "6"],
        hint="Repeatedly take total += n % base and divide n //= base until n becomes 0.",
        solution="    total = 0\n    while n > 0:\n        total += n % base\n        n //= base\n    return total",
        test="assert sum_of_digits_base(10, 2) == 2\nassert sum_of_digits_base(42, 10) == 6\nassert sum_of_digits_base(0, 10) == 0\nassert sum_of_digits_base(255, 16) == 30\nassert sum_of_digits_base(15, 8) == 8\n"
    ))

    # 30. Count Set Bits
    tasks.append(create_task(
        name="count_set_bits",
        signature="def count_set_bits(n: int) -> int:",
        doc_desc="Returns the count of set bits (1s) in the binary representation of integer n >= 0.",
        doctests=[">>> count_set_bits(7)", "3", ">>> count_set_bits(16)", "1"],
        hint="Use Brian Kernighan's algorithm: while n > 0: n &= n - 1; count += 1, or bin(n).count('1').",
        solution="    count = 0\n    while n > 0:\n        n &= (n - 1)\n        count += 1\n    return count",
        test="assert count_set_bits(7) == 3\nassert count_set_bits(16) == 1\nassert count_set_bits(0) == 0\nassert count_set_bits(15) == 4\nassert count_set_bits(1023) == 10\n"
    ))

    # 31. Next Power of Two
    tasks.append(create_task(
        name="next_power_of_two",
        signature="def next_power_of_two(n: int) -> int:",
        doc_desc="Finds the smallest power of 2 greater than or equal to non-negative integer n.",
        doctests=[">>> next_power_of_two(5)", "8", ">>> next_power_of_two(8)", "8"],
        hint="If n <= 1, return 1. Otherwise compute 1 << (n - 1).bit_length().",
        solution="    if n <= 1:\n        return 1\n    return 1 << (n - 1).bit_length()",
        test="assert next_power_of_two(5) == 8\nassert next_power_of_two(8) == 8\nassert next_power_of_two(0) == 1\nassert next_power_of_two(1) == 1\nassert next_power_of_two(17) == 32\n"
    ))

    # 32. Is Harshad Number
    tasks.append(create_task(
        name="is_harshad_number",
        signature="def is_harshad_number(n: int) -> bool:",
        doc_desc="Checks if positive integer n is divisible by the sum of its decimal digits.",
        doctests=[">>> is_harshad_number(18)", "True", ">>> is_harshad_number(19)", "False"],
        hint="If n <= 0 return False. Compute digit_sum = sum(int(d) for d in str(n)), and return n % digit_sum == 0.",
        solution="    if n <= 0:\n        return False\n    d_sum = sum(int(d) for d in str(n))\n    return n % d_sum == 0",
        test="assert is_harshad_number(18) is True\nassert is_harshad_number(19) is False\nassert is_harshad_number(21) is True\nassert is_harshad_number(0) is False\nassert is_harshad_number(100) is True\n"
    ))

    # 33. Is Automorphic Number
    tasks.append(create_task(
        name="is_automorphic_number",
        signature="def is_automorphic_number(n: int) -> bool:",
        doc_desc="Checks if the square of non-negative integer n ends with the digits of n.",
        doctests=[">>> is_automorphic_number(5)", "True", ">>> is_automorphic_number(7)", "False"],
        hint="Convert str(n * n) and check if it ends with str(n) using str.endswith().",
        solution="    if n < 0:\n        return False\n    return str(n * n).endswith(str(n))",
        test="assert is_automorphic_number(5) is True\nassert is_automorphic_number(25) is True\nassert is_automorphic_number(76) is True\nassert is_automorphic_number(7) is False\nassert is_automorphic_number(0) is True\nassert is_automorphic_number(1) is True\n"
    ))

    # 34. Triangular Number
    tasks.append(create_task(
        name="triangular_number",
        signature="def triangular_number(n: int) -> int:",
        doc_desc="Computes the nth triangular number T(n) = n * (n + 1) // 2.",
        doctests=[">>> triangular_number(4)", "10", ">>> triangular_number(1)", "1"],
        hint="Use the formula n * (n + 1) // 2 for n >= 0.",
        solution="    return n * (n + 1) // 2",
        test="assert triangular_number(4) == 10\nassert triangular_number(1) == 1\nassert triangular_number(0) == 0\nassert triangular_number(5) == 15\nassert triangular_number(10) == 55\n"
    ))

    # 35. Pentagonal Number
    tasks.append(create_task(
        name="pentagonal_number",
        signature="def pentagonal_number(n: int) -> int:",
        doc_desc="Computes the nth pentagonal number P(n) = n * (3n - 1) // 2.",
        doctests=[">>> pentagonal_number(1)", "1", ">>> pentagonal_number(5)", "35"],
        hint="Direct formula: n * (3 * n - 1) // 2.",
        solution="    return n * (3 * n - 1) // 2",
        test="assert pentagonal_number(1) == 1\nassert pentagonal_number(2) == 5\nassert pentagonal_number(3) == 12\nassert pentagonal_number(4) == 22\nassert pentagonal_number(5) == 35\n"
    ))

    # 36. Tribonacci Number
    tasks.append(create_task(
        name="tribonacci_number",
        signature="def tribonacci_number(n: int) -> int:",
        doc_desc="Computes the nth Tribonacci number T(0)=0, T(1)=1, T(2)=1, T(n)=T(n-1)+T(n-2)+T(n-3).",
        doctests=[">>> tribonacci_number(4)", "4", ">>> tribonacci_number(0)", "0"],
        hint="Initialize a, b, c = 0, 1, 1. Advance (a, b, c = b, c, a + b + c) n times.",
        solution="    if n == 0:\n        return 0\n    if n in (1, 2):\n        return 1\n    a, b, c = 0, 1, 1\n    for _ in range(3, n + 1):\n        a, b, c = b, c, a + b + c\n    return c",
        test="assert tribonacci_number(0) == 0\nassert tribonacci_number(1) == 1\nassert tribonacci_number(2) == 1\nassert tribonacci_number(4) == 4\nassert tribonacci_number(5) == 7\nassert tribonacci_number(6) == 13\n"
    ))

    # 37. Lucas Number
    tasks.append(create_task(
        name="lucas_number",
        signature="def lucas_number(n: int) -> int:",
        doc_desc="Computes the nth Lucas number: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2).",
        doctests=[">>> lucas_number(0)", "2", ">>> lucas_number(3)", "4"],
        hint="Track two variables starting at a, b = 2, 1. Advance a, b = b, a + b for n iterations.",
        solution="    if n == 0:\n        return 2\n    a, b = 2, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b",
        test="assert lucas_number(0) == 2\nassert lucas_number(1) == 1\nassert lucas_number(2) == 3\nassert lucas_number(3) == 4\nassert lucas_number(4) == 7\nassert lucas_number(5) == 11\n"
    ))

    # 38. Euler Totient
    tasks.append(create_task(
        name="euler_totient",
        signature="def euler_totient(n: int) -> int:",
        doc_desc="Computes Euler's totient phi(n): count of positive integers up to n coprime to n.",
        doctests=[">>> euler_totient(9)", "6", ">>> euler_totient(1)", "1"],
        hint="Start with result = n. For each prime p dividing n, multiply result *= (1 - 1/p) (using integer math result = result // p * (p - 1)).",
        solution="    result = n\n    p = 2\n    while p * p <= n:\n        if n % p == 0:\n            while n % p == 0:\n                n //= p\n            result = result // p * (p - 1)\n        p += 1\n    if n > 1:\n        result = result // n * (n - 1)\n    return result",
        test="assert euler_totient(9) == 6\nassert euler_totient(1) == 1\nassert euler_totient(10) == 4\nassert euler_totient(13) == 12\nassert euler_totient(36) == 12\n"
    ))

    # 39. Padovan Sequence
    tasks.append(create_task(
        name="padovan_number",
        signature="def padovan_number(n: int) -> int:",
        doc_desc="Computes the nth Padovan number P(n): P(0)=P(1)=P(2)=1, P(n)=P(n-2)+P(n-3).",
        doctests=[">>> padovan_number(3)", "2", ">>> padovan_number(5)", "3"],
        hint="Track three elements p0, p1, p2 = 1, 1, 1. For each step from 3 to n, next_p = p1 + p0; shift p0, p1, p2 = p1, p2, next_p.",
        solution="    if n in (0, 1, 2):\n        return 1\n    p0, p1, p2 = 1, 1, 1\n    for _ in range(3, n + 1):\n        p0, p1, p2 = p1, p2, p0 + p1\n    return p2",
        test="assert padovan_number(0) == 1\nassert padovan_number(3) == 2\nassert padovan_number(5) == 3\nassert padovan_number(8) == 7\nassert padovan_number(10) == 12\n"
    ))

    # 40. Sum of Multiples of 3 and 5
    tasks.append(create_task(
        name="sum_multiples_3_and_5",
        signature="def sum_multiples_3_and_5(limit: int) -> int:",
        doc_desc="Finds the sum of all natural numbers strictly below limit that are multiples of 3 or 5.",
        doctests=[">>> sum_multiples_3_and_5(10)", "23"],
        hint="Iterate over range(limit) and sum i if i % 3 == 0 or i % 5 == 0.",
        solution="    return sum(i for i in range(limit) if i % 3 == 0 or i % 5 == 0)",
        test="assert sum_multiples_3_and_5(10) == 23\nassert sum_multiples_3_and_5(16) == 60\nassert sum_multiples_3_and_5(1) == 0\nassert sum_multiples_3_and_5(0) == 0\n"
    ))

    # 41. Square Free Integer
    tasks.append(create_task(
        name="is_square_free",
        signature="def is_square_free(n: int) -> bool:",
        doc_desc="Determines if positive integer n is square-free (not divisible by any perfect square > 1).",
        doctests=[">>> is_square_free(10)", "True", ">>> is_square_free(18)", "False"],
        hint="Factor out primes: for d starting from 2, if n % (d * d) == 0 return False; while n % d == 0: n //= d. Stop when d*d > n.",
        solution="    if n <= 0:\n        return False\n    d = 2\n    while d * d <= n:\n        if n % (d * d) == 0:\n            return False\n        while n % d == 0:\n            n //= d\n        d += 1\n    return True",
        test="assert is_square_free(10) is True\nassert is_square_free(18) is False\nassert is_square_free(1) is True\nassert is_square_free(30) is True\nassert is_square_free(12) is False\n"
    ))

    # 42. Count Divisors
    tasks.append(create_task(
        name="count_divisors",
        signature="def count_divisors(n: int) -> int:",
        doc_desc="Returns the total number of positive divisors of integer n >= 1.",
        doctests=[">>> count_divisors(12)", "6", ">>> count_divisors(1)", "1"],
        hint="Loop d from 1 to sqrt(n). If n % d == 0, add 1 if d * d == n else 2.",
        solution="    if n <= 0:\n        return 0\n    count = 0\n    for d in range(1, int(n**0.5) + 1):\n        if n % d == 0:\n            count += 1 if d * d == n else 2\n    return count",
        test="assert count_divisors(12) == 6\nassert count_divisors(1) == 1\nassert count_divisors(7) == 2\nassert count_divisors(16) == 5\nassert count_divisors(100) == 9\n"
    ))

    # 43. Is Ugly Number
    tasks.append(create_task(
        name="is_ugly_number",
        signature="def is_ugly_number(n: int) -> bool:",
        doc_desc="Checks if positive integer n has only 2, 3, 5 as prime factors.",
        doctests=[">>> is_ugly_number(6)", "True", ">>> is_ugly_number(14)", "False"],
        hint="If n <= 0 return False. Divide out 2, 3, and 5 repeatedly; return n == 1.",
        solution="    if n <= 0:\n        return False\n    for p in (2, 3, 5):\n        while n % p == 0:\n            n //= p\n    return n == 1",
        test="assert is_ugly_number(6) is True\nassert is_ugly_number(1) is True\nassert is_ugly_number(14) is False\nassert is_ugly_number(0) is False\nassert is_ugly_number(30) is True\n"
    ))

    # 44. Is Happy Number
    tasks.append(create_task(
        name="is_happy_number",
        signature="def is_happy_number(n: int) -> bool:",
        doc_desc="Checks if n is a happy number (reaches 1 by repeatedly replacing n with sum of squares of digits).",
        doctests=[">>> is_happy_number(19)", "True", ">>> is_happy_number(2)", "False"],
        hint="Use a seen set to detect cycles: seen.add(n), compute n = sum(int(d)**2 for d in str(n)). If n == 1 return True.",
        solution="    seen = set()\n    while n != 1 and n not in seen:\n        seen.add(n)\n        n = sum(int(d)**2 for d in str(n))\n    return n == 1",
        test="assert is_happy_number(19) is True\nassert is_happy_number(2) is False\nassert is_happy_number(1) is True\nassert is_happy_number(7) is True\nassert is_happy_number(4) is False\n"
    ))

    # 45. Angle Between Clock Hands
    tasks.append(create_task(
        name="angle_between_clock_hands",
        signature="def angle_between_clock_hands(hour: int, minutes: int) -> float:",
        doc_desc="Calculates the smaller angle in degrees between hour and minute hands.",
        doctests=[">>> angle_between_clock_hands(12, 30)", "165.0", ">>> angle_between_clock_hands(3, 30)", "75.0"],
        hint="Hour hand position: (hour % 12) * 30 + minutes * 0.5. Minute hand: minutes * 6. Diff = abs(h - m). Return min(diff, 360 - diff).",
        solution="    h_angle = (hour % 12) * 30.0 + minutes * 0.5\n    m_angle = minutes * 6.0\n    diff = abs(h_angle - m_angle)\n    return min(diff, 360.0 - diff)",
        test="assert angle_between_clock_hands(12, 30) == 165.0\nassert angle_between_clock_hands(3, 30) == 75.0\nassert angle_between_clock_hands(3, 15) == 7.5\nassert angle_between_clock_hands(12, 0) == 0.0\n"
    ))

    # 46. Is Leap Year
    tasks.append(create_task(
        name="is_leap_year",
        signature="def is_leap_year(year: int) -> bool:",
        doc_desc="Determines if a given Gregorian calendar year is a leap year.",
        doctests=[">>> is_leap_year(2000)", "True", ">>> is_leap_year(1900)", "False"],
        hint="A year is leap if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0).",
        solution="    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)",
        test="assert is_leap_year(2000) is True\nassert is_leap_year(1900) is False\nassert is_leap_year(2024) is True\nassert is_leap_year(2023) is False\n"
    ))

    # 47. Manhattan Distance 2D
    tasks.append(create_task(
        name="manhattan_distance_2d",
        signature="def manhattan_distance_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:",
        doc_desc="Calculates Manhattan (L1) distance between two 2D points.",
        doctests=[">>> manhattan_distance_2d((0, 0), (3, 4))", "7.0"],
        hint="Sum the absolute differences: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]).",
        solution="    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])",
        test="assert manhattan_distance_2d((0, 0), (3, 4)) == 7.0\nassert manhattan_distance_2d((-1, -1), (2, 3)) == 7.0\nassert manhattan_distance_2d((5, 5), (5, 5)) == 0.0\n"
    ))

    # 48. Euclidean Distance 2D
    tasks.append(create_task(
        name="euclidean_distance_2d",
        signature="def euclidean_distance_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:",
        doc_desc="Calculates Euclidean (L2) distance between two 2D points.",
        doctests=[">>> euclidean_distance_2d((0, 0), (3, 4))", "5.0"],
        hint="Use math.hypot(p1[0] - p2[0], p1[1] - p2[1]).",
        solution="    import math\n    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])",
        test="assert abs(euclidean_distance_2d((0, 0), (3, 4)) - 5.0) < 1e-6\nassert abs(euclidean_distance_2d((1, 1), (1, 1)) - 0.0) < 1e-6\nassert abs(euclidean_distance_2d((1, 2), (4, 6)) - 5.0) < 1e-6\n"
    ))

    # 49. Dot Product Vectors
    tasks.append(create_task(
        name="dot_product_vectors",
        signature="def dot_product_vectors(v1: list[float], v2: list[float]) -> float:",
        doc_desc="Calculates the dot product of two vectors of equal length.",
        doctests=[">>> dot_product_vectors([1.0, 2.0], [3.0, 4.0])", "11.0"],
        hint="Sum products across pairs: sum(a * b for a, b in zip(v1, v2)). Raise ValueError if lengths differ.",
        solution="    if len(v1) != len(v2):\n        raise ValueError('Vectors must have equal length')\n    return sum(a * b for a, b in zip(v1, v2))",
        test="assert dot_product_vectors([1.0, 2.0], [3.0, 4.0]) == 11.0\nassert dot_product_vectors([], []) == 0.0\nassert dot_product_vectors([1.0, 0.0], [0.0, 1.0]) == 0.0\n"
    ))

    # 50. Cross Product 2D
    tasks.append(create_task(
        name="cross_product_2d",
        signature="def cross_product_2d(v1: tuple[float, float], v2: tuple[float, float]) -> float:",
        doc_desc="Computes the 2D cross product magnitude (v1[0]*v2[1] - v1[1]*v2[0]).",
        doctests=[">>> cross_product_2d((1, 0), (0, 1))", "1.0"],
        hint="Direct formula: v1[0] * v2[1] - v1[1] * v2[0].",
        solution="    return v1[0] * v2[1] - v1[1] * v2[0]",
        test="assert cross_product_2d((1, 0), (0, 1)) == 1.0\nassert cross_product_2d((2, 3), (4, 6)) == 0.0\nassert cross_product_2d((1, 2), (3, 4)) == -2.0\n"
    ))

    # 51. Are Points Collinear
    tasks.append(create_task(
        name="are_points_collinear",
        signature="def are_points_collinear(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> bool:",
        doc_desc="Determines whether three 2D points lie on the same straight line.",
        doctests=[">>> are_points_collinear((0, 0), (1, 1), (2, 2))", "True", ">>> are_points_collinear((0, 0), (1, 1), (1, 2))", "False"],
        hint="Compute cross product area: (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]); check if abs(area) < 1e-9.",
        solution="    area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])\n    return abs(area) < 1e-9",
        test="assert are_points_collinear((0, 0), (1, 1), (2, 2)) is True\nassert are_points_collinear((0, 0), (1, 1), (1, 2)) is False\nassert are_points_collinear((1, 2), (3, 4), (5, 6)) is True\n"
    ))

    # 52. Round to Nearest Multiple
    tasks.append(create_task(
        name="round_to_nearest_multiple",
        signature="def round_to_nearest_multiple(n: int, m: int) -> int:",
        doc_desc="Rounds integer n to the nearest multiple of m (half rounds away from zero).",
        doctests=[">>> round_to_nearest_multiple(23, 5)", "25", ">>> round_to_nearest_multiple(22, 5)", "20"],
        hint="Compute rem = n % m. If rem >= (m + 1) // 2 return n + (m - rem); else return n - rem.",
        solution="    if m <= 0:\n        raise ValueError('m must be positive')\n    rem = n % m\n    if rem >= (m + 1) // 2:\n        return n + (m - rem)\n    return n - rem",
        test="assert round_to_nearest_multiple(23, 5) == 25\nassert round_to_nearest_multiple(22, 5) == 20\nassert round_to_nearest_multiple(15, 5) == 15\nassert round_to_nearest_multiple(10, 4) == 12\n"
    ))

    # 53. Is Abundant Number
    tasks.append(create_task(
        name="is_abundant_number",
        signature="def is_abundant_number(n: int) -> bool:",
        doc_desc="Checks if sum of proper divisors of positive integer n is strictly greater than n.",
        doctests=[">>> is_abundant_number(12)", "True", ">>> is_abundant_number(15)", "False"],
        hint="Calculate proper divisor sum. Return sum_divisors > n for n > 0.",
        solution="    if n <= 1:\n        return False\n    total = 1\n    for d in range(2, int(n**0.5) + 1):\n        if n % d == 0:\n            total += d\n            if d != n // d:\n                total += n // d\n    return total > n",
        test="assert is_abundant_number(12) is True\nassert is_abundant_number(15) is False\nassert is_abundant_number(18) is True\nassert is_abundant_number(1) is False\n"
    ))

    # 54. Is Deficient Number
    tasks.append(create_task(
        name="is_deficient_number",
        signature="def is_deficient_number(n: int) -> bool:",
        doc_desc="Checks if sum of proper divisors of positive integer n is strictly less than n.",
        doctests=[">>> is_deficient_number(15)", "True", ">>> is_deficient_number(12)", "False"],
        hint="Calculate proper divisor sum. Return sum_divisors < n for n >= 1.",
        solution="    if n <= 0:\n        return False\n    if n == 1:\n        return True\n    total = 1\n    for d in range(2, int(n**0.5) + 1):\n        if n % d == 0:\n            total += d\n            if d != n // d:\n                total += n // d\n    return total < n",
        test="assert is_deficient_number(15) is True\nassert is_deficient_number(12) is False\nassert is_deficient_number(1) is True\nassert is_deficient_number(7) is True\n"
    ))

    # 55. Aliquot Sum
    tasks.append(create_task(
        name="aliquot_sum",
        signature="def aliquot_sum(n: int) -> int:",
        doc_desc="Returns the sum of all proper divisors of positive integer n.",
        doctests=[">>> aliquot_sum(15)", "9", ">>> aliquot_sum(6)", "6"],
        hint="Proper divisors are positive divisors excluding n. Return 0 for n <= 1.",
        solution="    if n <= 1:\n        return 0\n    total = 1\n    for d in range(2, int(n**0.5) + 1):\n        if n % d == 0:\n            total += d\n            if d != n // d:\n                total += n // d\n    return total",
        test="assert aliquot_sum(15) == 9\nassert aliquot_sum(6) == 6\nassert aliquot_sum(1) == 0\nassert aliquot_sum(28) == 28\n"
    ))

    # 56. Collatz Max Value
    tasks.append(create_task(
        name="collatz_max_value",
        signature="def collatz_max_value(n: int) -> int:",
        doc_desc="Returns the peak (maximum) integer reached during the Collatz sequence starting at n.",
        doctests=[">>> collatz_max_value(3)", "16", ">>> collatz_max_value(1)", "1"],
        hint="Track max_val = n. While n > 1: n = n // 2 if n % 2 == 0 else 3 * n + 1; max_val = max(max_val, n).",
        solution="    max_val = n\n    while n > 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        if n > max_val:\n            max_val = n\n    return max_val",
        test="assert collatz_max_value(3) == 16\nassert collatz_max_value(1) == 1\nassert collatz_max_value(7) == 52\nassert collatz_max_value(6) == 16\n"
    ))

    # 57. Binary Gap Max Zeroes
    tasks.append(create_task(
        name="binary_gap_max_zeros",
        signature="def binary_gap_max_zeros(n: int) -> int:",
        doc_desc="Finds the length of the longest sequence of consecutive zeros surrounded by 1s in binary representation of n.",
        doctests=[">>> binary_gap_max_zeros(9)", "2", ">>> binary_gap_max_zeros(529)", "4"],
        hint="Convert to bin(n)[2:].strip('0'). Split by '1' and return max length of segments (or 0 if no gaps).",
        solution="    b = bin(n)[2:].strip('0')\n    gaps = b.split('1')\n    return max((len(g) for g in gaps), default=0)",
        test="assert binary_gap_max_zeros(9) == 2\nassert binary_gap_max_zeros(529) == 4\nassert binary_gap_max_zeros(15) == 0\nassert binary_gap_max_zeros(32) == 0\n"
    ))

    # 58. Sum of Squares of First N
    tasks.append(create_task(
        name="sum_of_squares_first_n",
        signature="def sum_of_squares_first_n(n: int) -> int:",
        doc_desc="Calculates the sum of squares of the first n positive integers 1^2 + 2^2 + ... + n^2.",
        doctests=[">>> sum_of_squares_first_n(3)", "14", ">>> sum_of_squares_first_n(0)", "0"],
        hint="Use the formula n * (n + 1) * (2 * n + 1) // 6.",
        solution="    return n * (n + 1) * (2 * n + 1) // 6",
        test="assert sum_of_squares_first_n(3) == 14\nassert sum_of_squares_first_n(0) == 0\nassert sum_of_squares_first_n(1) == 1\nassert sum_of_squares_first_n(5) == 55\n"
    ))

    # 59. Square Pyramidal Number
    tasks.append(create_task(
        name="square_pyramidal_number",
        signature="def square_pyramidal_number(n: int) -> int:",
        doc_desc="Computes the nth square pyramidal number P(n) = n * (n + 1) * (2n + 1) // 6.",
        doctests=[">>> square_pyramidal_number(2)", "5", ">>> square_pyramidal_number(4)", "30"],
        hint="Equivalent to the sum of the first n squares: n * (n + 1) * (2 * n + 1) // 6.",
        solution="    return n * (n + 1) * (2 * n + 1) // 6",
        test="assert square_pyramidal_number(2) == 5\nassert square_pyramidal_number(4) == 30\nassert square_pyramidal_number(1) == 1\nassert square_pyramidal_number(0) == 0\n"
    ))

    # 60. Is Kaprekar Number
    tasks.append(create_task(
        name="is_kaprekar_number",
        signature="def is_kaprekar_number(n: int) -> bool:",
        doc_desc="Determines if positive integer n is a Kaprekar number (n^2 split into two parts sums to n).",
        doctests=[">>> is_kaprekar_number(45)", "True", ">>> is_kaprekar_number(9)", "True", ">>> is_kaprekar_number(10)", "False"],
        hint="Square n. Split string of sq into right part (length len(str(n))) and left part. If int(left) + int(right) == n and int(right) > 0, return True.",
        solution="    if n == 1:\n        return True\n    sq = n * n\n    s = str(sq)\n    d = len(str(n))\n    right_str = s[-d:]\n    left_str = s[:-d]\n    left = int(left_str) if left_str else 0\n    right = int(right_str) if right_str else 0\n    return right > 0 and (left + right == n)",
        test="assert is_kaprekar_number(45) is True\nassert is_kaprekar_number(9) is True\nassert is_kaprekar_number(1) is True\nassert is_kaprekar_number(10) is False\nassert is_kaprekar_number(297) is True\n"
    ))

    # 61. Sum of Cube of First N
    tasks.append(create_task(
        name="sum_of_cubes_first_n",
        signature="def sum_of_cubes_first_n(n: int) -> int:",
        doc_desc="Calculates 1^3 + 2^3 + ... + n^3 using the closed form identity (n * (n + 1) // 2) ** 2.",
        doctests=[">>> sum_of_cubes_first_n(3)", "36", ">>> sum_of_cubes_first_n(1)", "1"],
        hint="Direct formula: (n * (n + 1) // 2) ** 2.",
        solution="    return (n * (n + 1) // 2) ** 2",
        test="assert sum_of_cubes_first_n(3) == 36\nassert sum_of_cubes_first_n(1) == 1\nassert sum_of_cubes_first_n(0) == 0\nassert sum_of_cubes_first_n(4) == 100\n"
    ))

    # 62. Centered Polygonal Hexagonal
    tasks.append(create_task(
        name="hexagonal_number",
        signature="def hexagonal_number(n: int) -> int:",
        doc_desc="Computes the nth hexagonal number H(n) = n * (2n - 1).",
        doctests=[">>> hexagonal_number(1)", "1", ">>> hexagonal_number(4)", "28"],
        hint="Use the standard formula n * (2 * n - 1).",
        solution="    return n * (2 * n - 1)",
        test="assert hexagonal_number(1) == 1\nassert hexagonal_number(4) == 28\nassert hexagonal_number(2) == 6\nassert hexagonal_number(3) == 15\n"
    ))

    # 63. Fast Modular Power
    tasks.append(create_task(
        name="fast_modular_power",
        signature="def fast_modular_power(a: int, b: int, m: int) -> int:",
        doc_desc="Calculates (a ** b) % m without using the 3-argument pow built-in.",
        doctests=[">>> fast_modular_power(2, 5, 13)", "6"],
        hint="Use square-and-multiply: res = 1; while b > 0: if b & 1: res = (res * a) % m; a = (a * a) % m; b >>= 1.",
        solution="    res = 1\n    a %= m\n    while b > 0:\n        if b & 1:\n            res = (res * a) % m\n        a = (a * a) % m\n        b >>= 1\n    return res",
        test="assert fast_modular_power(2, 5, 13) == 6\nassert fast_modular_power(3, 4, 100) == 81\nassert fast_modular_power(5, 0, 7) == 1\n"
    ))

    # 64. Count Permutations with Duplicates
    tasks.append(create_task(
        name="count_unique_permutations_multiset",
        signature="def count_unique_permutations_multiset(elements: list[int]) -> int:",
        doc_desc="Counts the number of distinct permutations of a list containing duplicate integers.",
        doctests=[">>> count_unique_permutations_multiset([1, 1, 2])", "3"],
        hint="Compute len(elements)! divided by the product of count! for each unique element frequency using math.factorial and collections.Counter.",
        solution="    import math\n    from collections import Counter\n    counts = Counter(elements)\n    denom = 1\n    for c in counts.values():\n        denom *= math.factorial(c)\n    return math.factorial(len(elements)) // denom",
        test="assert count_unique_permutations_multiset([1, 1, 2]) == 3\nassert count_unique_permutations_multiset([1, 2, 3]) == 6\nassert count_unique_permutations_multiset([1, 1, 1, 1]) == 1\nassert count_unique_permutations_multiset([]) == 1\n"
    ))

    return tasks
