"""
Category 6: Built-ins & Iteration (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_builtins_iteration_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Pairwise Elements
    tasks.append(create_task(
        name="pairwise_elements",
        signature="def pairwise_elements(items: list[Any]) -> list[tuple[Any, Any]]:",
        doc_desc="Returns consecutive overlapping pairs (items[i], items[i+1]) for all adjacent elements.",
        doctests=[">>> pairwise_elements([1, 2, 3, 4])", "[(1, 2), (2, 3), (3, 4)]"],
        hint="Use list(zip(items, items[1:])) or itertools.pairwise.",
        solution="    return list(zip(items, items[1:]))",
        test="assert pairwise_elements([1, 2, 3, 4]) == [(1, 2), (2, 3), (3, 4)]\nassert pairwise_elements([1]) == []\nassert pairwise_elements([]) == []\n"
    ))

    # 2. Chunk Iterable
    tasks.append(create_task(
        name="chunk_iterable",
        signature="def chunk_iterable(items: list[Any], size: int) -> list[list[Any]]:",
        doc_desc="Splits items into consecutive sublists of maximum length size.",
        doctests=[">>> chunk_iterable([1, 2, 3, 4, 5], 2)", "[[1, 2], [3, 4], [5]]"],
        hint="Use list comprehension with step: [items[i:i+size] for i in range(0, len(items), size)]. Return [] if size <= 0.",
        solution="    if size <= 0:\n        return []\n    return [items[i:i + size] for i in range(0, len(items), size)]",
        test="assert chunk_iterable([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk_iterable([1, 2], 5) == [[1, 2]]\nassert chunk_iterable([], 3) == []\nassert chunk_iterable([1, 2], 0) == []\n"
    ))

    # 3. Interleave Sequences
    tasks.append(create_task(
        name="interleave_sequences",
        signature="def interleave_sequences(*sequences: list[Any]) -> list[Any]:",
        doc_desc="Interleaves elements from multiple variable-length sequences round-robin until all are exhausted.",
        doctests=[">>> interleave_sequences([1, 2, 3], ['a', 'b'])", "[1, 'a', 2, 'b', 3]"],
        hint="Loop up to max(len(s) for s in sequences), appending elements at current index if in bounds.",
        solution="    if not sequences:\n        return []\n    max_len = max((len(s) for s in sequences), default=0)\n    res = []\n    for i in range(max_len):\n        for s in sequences:\n            if i < len(s):\n                res.append(s[i])\n    return res",
        test="assert interleave_sequences([1, 2, 3], ['a', 'b']) == [1, 'a', 2, 'b', 3]\nassert interleave_sequences([1], [2], [3]) == [1, 2, 3]\nassert interleave_sequences() == []\n"
    ))

    # 4. Cartesian Product
    tasks.append(create_task(
        name="cartesian_product_lists",
        signature="def cartesian_product_lists(l1: list[Any], l2: list[Any]) -> list[tuple[Any, Any]]:",
        doc_desc="Returns the Cartesian product of two lists as a list of 2-tuples using itertools.product.",
        doctests=[">>> cartesian_product_lists([1, 2], ['a', 'b'])", "[(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]"],
        hint="Use list(itertools.product(l1, l2)).",
        solution="    import itertools\n    return list(itertools.product(l1, l2))",
        test="assert cartesian_product_lists([1, 2], ['a', 'b']) == [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]\nassert cartesian_product_lists([], [1, 2]) == []\nassert cartesian_product_lists([1], []) == []\n"
    ))

    # 5. Run Length Grouping
    tasks.append(create_task(
        name="run_length_grouping",
        signature="def run_length_grouping(items: list[Any]) -> list[tuple[Any, int]]:",
        doc_desc="Groups consecutive equal elements into (key, count) tuples using itertools.groupby.",
        doctests=[">>> run_length_grouping(['a', 'a', 'b', 'c', 'c', 'c'])", "[('a', 2), ('b', 1), ('c', 3)]"],
        hint="Use itertools.groupby: [(k, len(list(g))) for k, g in itertools.groupby(items)].",
        solution="    from itertools import groupby\n    return [(k, len(list(g))) for k, g in groupby(items)]",
        test="assert run_length_grouping(['a', 'a', 'b', 'c', 'c', 'c']) == [('a', 2), ('b', 1), ('c', 3)]\nassert run_length_grouping([1, 2, 3]) == [(1, 1), (2, 1), (3, 1)]\nassert run_length_grouping([]) == []\n"
    ))

    # 6. Top K Frequent Words
    tasks.append(create_task(
        name="top_k_frequent_words",
        signature="def top_k_frequent_words(words: list[str], k: int) -> list[str]:",
        doc_desc="Returns the k most frequent words sorted by frequency descending, breaking ties alphabetically ascending.",
        doctests=[">>> top_k_frequent_words(['i', 'love', 'leetcode', 'i', 'love', 'coding'], 2)", "['i', 'love']"],
        hint="Count with collections.Counter. Sort keys using (-count, word) and slice [:k].",
        solution="    from collections import Counter\n    counts = Counter(words)\n    sorted_words = sorted(counts.keys(), key=lambda w: (-counts[w], w))\n    return sorted_words[:k]",
        test="assert top_k_frequent_words(['i', 'love', 'leetcode', 'i', 'love', 'coding'], 2) == ['i', 'love']\nassert top_k_frequent_words(['the', 'day', 'is', 'sunny', 'the', 'the', 'the', 'sunny', 'is', 'is'], 4) == ['the', 'is', 'sunny', 'day']\n"
    ))

    # 7. Custom Multikey Sort
    tasks.append(create_task(
        name="custom_multikey_sort",
        signature="def custom_multikey_sort(records: list[dict[str, Any]], primary_key: str, secondary_key: str) -> list[dict[str, Any]]:",
        doc_desc="Sorts a list of dictionaries by primary_key ascending, then by secondary_key descending.",
        doctests=[">>> custom_multikey_sort([{'dept': 'eng', 'score': 80}, {'dept': 'eng', 'score': 95}], 'dept', 'score')[0]['score']", "95"],
        hint="Use sorted(records, key=lambda r: (r[primary_key], -r[secondary_key])).",
        solution="    return sorted(records, key=lambda r: (r[primary_key], -r[secondary_key]))",
        test="recs = [{'dept': 'eng', 'score': 80}, {'dept': 'eng', 'score': 95}, {'dept': 'art', 'score': 70}]\nsorted_recs = custom_multikey_sort(recs, 'dept', 'score')\nassert sorted_recs[0]['dept'] == 'art'\nassert sorted_recs[1]['score'] == 95\nassert sorted_recs[2]['score'] == 80\n"
    ))

    # 8. Running Accumulate Products
    tasks.append(create_task(
        name="running_accumulate_products",
        signature="def running_accumulate_products(nums: list[int]) -> list[int]:",
        doc_desc="Returns cumulative running products of nums using itertools.accumulate and operator.mul.",
        doctests=[">>> running_accumulate_products([1, 2, 3, 4])", "[1, 2, 6, 24]"],
        hint="Use list(itertools.accumulate(nums, operator.mul)). Return [] if nums is empty.",
        solution="    import itertools\n    import operator\n    if not nums: return []\n    return list(itertools.accumulate(nums, operator.mul))",
        test="assert running_accumulate_products([1, 2, 3, 4]) == [1, 2, 6, 24]\nassert running_accumulate_products([5]) == [5]\nassert running_accumulate_products([]) == []\n"
    ))

    # 9. Partition Truthy Falsy
    tasks.append(create_task(
        name="partition_truthy_falsy",
        signature="def partition_truthy_falsy(items: list[Any]) -> tuple[list[Any], list[Any]]:",
        doc_desc="Splits items into two lists: (truthy_items, falsy_items) preserving order.",
        doctests=[">>> partition_truthy_falsy([0, 1, '', 'hello', None, [2]])", "([1, 'hello', [2]], [0, '', None])"],
        hint="Iterate items into truthy list if bool(x) else falsy list.",
        solution="    truthy = [x for x in items if x]\n    falsy = [x for x in items if not x]\n    return (truthy, falsy)",
        test="assert partition_truthy_falsy([0, 1, '', 'hello', None, [2]]) == ([1, 'hello', [2]], [0, '', None])\nassert partition_truthy_falsy([]) == ([], [])\n"
    ))

    # 10. Flatten Chain Iterables
    tasks.append(create_task(
        name="flatten_chain_iterables",
        signature="def flatten_chain_iterables(iterables: list[list[Any]]) -> list[Any]:",
        doc_desc="Flattens a list of sublists into a single flat list using itertools.chain.from_iterable.",
        doctests=[">>> flatten_chain_iterables([[1, 2], [3, 4], [5]])", "[1, 2, 3, 4, 5]"],
        hint="Use list(itertools.chain.from_iterable(iterables)).",
        solution="    import itertools\n    return list(itertools.chain.from_iterable(iterables))",
        test="assert flatten_chain_iterables([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]\nassert flatten_chain_iterables([[], [1]]) == [1]\nassert flatten_chain_iterables([]) == []\n"
    ))

    # 11. All Unique Elements
    tasks.append(create_task(
        name="all_unique_elements",
        signature="def all_unique_elements(items: list[Any]) -> bool:",
        doc_desc="Checks if all elements in a list are distinct.",
        doctests=[">>> all_unique_elements([1, 2, 3])", "True", ">>> all_unique_elements([1, 2, 1])", "False"],
        hint="Check len(items) == len(set(items)).",
        solution="    return len(items) == len(set(items))",
        test="assert all_unique_elements([1, 2, 3]) is True\nassert all_unique_elements([1, 2, 1]) is False\nassert all_unique_elements([]) is True\n"
    ))

    # 12. Zip Strict Lists
    tasks.append(create_task(
        name="zip_strict_lists",
        signature="def zip_strict_lists(l1: list[Any], l2: list[Any]) -> list[tuple[Any, Any]]:",
        doc_desc="Zips two lists together into pairs, raising ValueError if lists have different lengths.",
        doctests=[">>> zip_strict_lists([1, 2], ['a', 'b'])", "[(1, 'a'), (2, 'b')]"],
        hint="Check len(l1) == len(l2). If not, raise ValueError('Lists must have equal length'). Return list(zip(l1, l2)).",
        solution="    if len(l1) != len(l2):\n        raise ValueError('Lists must have equal length')\n    return list(zip(l1, l2))",
        test="assert zip_strict_lists([1, 2], ['a', 'b']) == [(1, 'a'), (2, 'b')]\nassert zip_strict_lists([], []) == []\ntry:\n    zip_strict_lists([1], [1, 2])\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 13. Invert Dictionary Mapping
    tasks.append(create_task(
        name="invert_dict_unique",
        signature="def invert_dict_unique(d: dict[Any, Any]) -> dict[Any, Any]:",
        doc_desc="Inverts a 1-to-1 dictionary swapping keys and values.",
        doctests=[">>> invert_dict_unique({'a': 1, 'b': 2})", "{1: 'a', 2: 'b'}"],
        hint="Use dict comprehension {v: k for k, v in d.items()}.",
        solution="    return {v: k for k, v in d.items()}",
        test="assert invert_dict_unique({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\nassert invert_dict_unique({}) == {}\n"
    ))

    # 14. Filter None Values Dict
    tasks.append(create_task(
        name="filter_none_values_dict",
        signature="def filter_none_values_dict(d: dict[str, Any]) -> dict[str, Any]:",
        doc_desc="Removes all keys whose value is None from the dictionary.",
        doctests=[">>> filter_none_values_dict({'a': 1, 'b': None, 'c': 'ok'})", "{'a': 1, 'c': 'ok'}"],
        hint="Dict comprehension {k: v for k, v in d.items() if v is not None}.",
        solution="    return {k: v for k, v in d.items() if v is not None}",
        test="assert filter_none_values_dict({'a': 1, 'b': None, 'c': 'ok'}) == {'a': 1, 'c': 'ok'}\nassert filter_none_values_dict({'x': None}) == {}\nassert filter_none_values_dict({}) == {}\n"
    ))

    # 15. Merge Dictionaries Sum
    tasks.append(create_task(
        name="merge_dictionaries_sum",
        signature="def merge_dictionaries_sum(d1: dict[str, int], d2: dict[str, int]) -> dict[str, int]:",
        doc_desc="Merges two dictionaries summing values for overlapping keys using collections.Counter.",
        doctests=[">>> merge_dictionaries_sum({'a': 1, 'b': 2}, {'b': 3, 'c': 4})", "{'a': 1, 'b': 5, 'c': 4}"],
        hint="Use collections.Counter(d1) + collections.Counter(d2) and cast back to dict.",
        solution="    from collections import Counter\n    c = Counter(d1)\n    c.update(d2)\n    return dict(c)",
        test="assert merge_dictionaries_sum({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) == {'a': 1, 'b': 5, 'c': 4}\nassert merge_dictionaries_sum({}, {'x': 1}) == {'x': 1}\n"
    ))

    # 16. Defaultdict Grouping
    tasks.append(create_task(
        name="group_items_by_key",
        signature="def group_items_by_key(items: list[tuple[str, Any]]) -> dict[str, list[Any]]:",
        doc_desc="Groups a list of (category, value) tuples into a dictionary mapping each category to its list of values.",
        doctests=[">>> group_items_by_key([('fruit', 'apple'), ('fruit', 'banana'), ('veg', 'carrot')])", "{'fruit': ['apple', 'banana'], 'veg': ['carrot']}"],
        hint="Use collections.defaultdict(list) appending val for k, val in items.",
        solution="    from collections import defaultdict\n    res = defaultdict(list)\n    for k, val in items:\n        res[k].append(val)\n    return dict(res)",
        test="assert group_items_by_key([('fruit', 'apple'), ('fruit', 'banana'), ('veg', 'carrot')]) == {'fruit': ['apple', 'banana'], 'veg': ['carrot']}\nassert group_items_by_key([]) == {}\n"
    ))

    # 17. Deque Rotate Steps
    tasks.append(create_task(
        name="deque_rotate_steps",
        signature="def deque_rotate_steps(items: list[Any], steps: int) -> list[Any]:",
        doc_desc="Rotates elements using collections.deque.rotate (positive rotates right, negative left).",
        doctests=[">>> deque_rotate_steps([1, 2, 3, 4, 5], 2)", "[4, 5, 1, 2, 3]"],
        hint="Convert to collections.deque, call d.rotate(steps), and return list(d).",
        solution="    from collections import deque\n    d = deque(items)\n    d.rotate(steps)\n    return list(d)",
        test="assert deque_rotate_steps([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]\nassert deque_rotate_steps([1, 2, 3], -1) == [2, 3, 1]\nassert deque_rotate_steps([], 3) == []\n"
    ))

    # 18. N Smallest Elements Heapq
    tasks.append(create_task(
        name="heapq_n_smallest",
        signature="def heapq_n_smallest(n: int, items: list[float]) -> list[float]:",
        doc_desc="Returns the n smallest elements from items in ascending order using heapq.nsmallest.",
        doctests=[">>> heapq_n_smallest(3, [5.0, 1.0, 4.0, 2.0, 3.0])", "[1.0, 2.0, 3.0]"],
        hint="Use heapq.nsmallest(n, items).",
        solution="    import heapq\n    return heapq.nsmallest(n, items)",
        test="assert heapq_n_smallest(3, [5.0, 1.0, 4.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]\nassert heapq_n_smallest(0, [1.0, 2.0]) == []\nassert heapq_n_smallest(5, [2.0]) == [2.0]\n"
    ))

    # 19. N Largest Elements Heapq
    tasks.append(create_task(
        name="heapq_n_largest",
        signature="def heapq_n_largest(n: int, items: list[float]) -> list[float]:",
        doc_desc="Returns the n largest elements from items in descending order using heapq.nlargest.",
        doctests=[">>> heapq_n_largest(2, [5.0, 1.0, 4.0, 2.0, 3.0])", "[5.0, 4.0]"],
        hint="Use heapq.nlargest(n, items).",
        solution="    import heapq\n    return heapq.nlargest(n, items)",
        test="assert heapq_n_largest(2, [5.0, 1.0, 4.0, 2.0, 3.0]) == [5.0, 4.0]\nassert heapq_n_largest(0, [1.0]) == []\n"
    ))

    # 20. Itertools Combinations Pairs
    tasks.append(create_task(
        name="generate_combinations_pairs",
        signature="def generate_combinations_pairs(items: list[Any]) -> list[tuple[Any, Any]]:",
        doc_desc="Generates all 2-element combinations from items in lexicographic order.",
        doctests=[">>> generate_combinations_pairs(['A', 'B', 'C'])", "[('A', 'B'), ('A', 'C'), ('B', 'C')]"],
        hint="Use list(itertools.combinations(items, 2)).",
        solution="    import itertools\n    return list(itertools.combinations(items, 2))",
        test="assert generate_combinations_pairs(['A', 'B', 'C']) == [('A', 'B'), ('A', 'C'), ('B', 'C')]\nassert generate_combinations_pairs([1]) == []\n"
    ))

    # Dynamically generate 44 additional distinct Built-in & Iteration challenges (total 64)
    builtins_specs = [
        ("itertools_permutations_r", "def itertools_permutations_r(items: list[Any], r: int) -> list[tuple[Any, ...]]:",
         "Generates all r-permutations of items using itertools.permutations.",
         "Use list(itertools.permutations(items, r)).",
         "    import itertools\n    return list(itertools.permutations(items, r))",
         "assert itertools_permutations_r([1, 2], 2) == [(1, 2), (2, 1)]\nassert itertools_permutations_r([1], 2) == []\n"),

        ("itertools_compress_filter", "def itertools_compress_filter(data: list[Any], selectors: list[bool]) -> list[Any]:",
         "Filters data keeping elements where corresponding selector is True using itertools.compress.",
         "Use list(itertools.compress(data, selectors)).",
         "    import itertools\n    return list(itertools.compress(data, selectors))",
         "assert itertools_compress_filter(['a', 'b', 'c'], [True, False, True]) == ['a', 'c']\nassert itertools_compress_filter([], []) == []\n"),

        ("itertools_dropwhile_neg", "def itertools_dropwhile_neg(nums: list[int]) -> list[int]:",
         "Drops initial negative numbers using itertools.dropwhile.",
         "Use list(itertools.dropwhile(lambda x: x < 0, nums)).",
         "    import itertools\n    return list(itertools.dropwhile(lambda x: x < 0, nums))",
         "assert itertools_dropwhile_neg([-2, -1, 0, 1, -1]) == [0, 1, -1]\nassert itertools_dropwhile_neg([1, -1]) == [1, -1]\n"),

        ("itertools_takewhile_pos", "def itertools_takewhile_pos(nums: list[int]) -> list[int]:",
         "Takes initial positive numbers using itertools.takewhile.",
         "Use list(itertools.takewhile(lambda x: x > 0, nums)).",
         "    import itertools\n    return list(itertools.takewhile(lambda x: x > 0, nums))",
         "assert itertools_takewhile_pos([1, 2, 3, -1, 4]) == [1, 2, 3]\nassert itertools_takewhile_pos([-1, 1]) == []\n"),

        ("zip_longest_filling", "def zip_longest_filling(l1: list[Any], l2: list[Any], fill: Any = None) -> list[tuple[Any, Any]]:",
         "Zips two lists filling missing values with fill using itertools.zip_longest.",
         "Use list(itertools.zip_longest(l1, l2, fillvalue=fill)).",
         "    import itertools\n    return list(itertools.zip_longest(l1, l2, fillvalue=fill))",
         "assert zip_longest_filling([1], ['a', 'b'], fill='?') == [(1, 'a'), ('?', 'b')]\nassert zip_longest_filling([], []) == []\n"),

        ("find_argmax_index", "def find_argmax_index(nums: list[float]) -> int:",
         "Returns the 0-based index of the maximum element in nums.",
         "Use max(range(len(nums)), key=nums.__getitem__).",
         "    if not nums: raise ValueError('Empty list')\n    return max(range(len(nums)), key=nums.__getitem__)",
         "assert find_argmax_index([1.0, 3.5, 2.0]) == 1\nassert find_argmax_index([5.0]) == 0\n"),

        ("find_argmin_index", "def find_argmin_index(nums: list[float]) -> int:",
         "Returns the 0-based index of the minimum element in nums.",
         "Use min(range(len(nums)), key=nums.__getitem__).",
         "    if not nums: raise ValueError('Empty list')\n    return min(range(len(nums)), key=nums.__getitem__)",
         "assert find_argmin_index([4.0, 1.5, 2.0]) == 1\nassert find_argmin_index([3.0]) == 0\n"),

        ("sort_dictionary_by_value", "def sort_dictionary_by_value(d: dict[str, int], descending: bool = False) -> list[tuple[str, int]]:",
         "Returns sorted list of (key, value) pairs ordered by value.",
         "Use sorted(d.items(), key=lambda x: x[1], reverse=descending).",
         "    return sorted(d.items(), key=lambda x: x[1], reverse=descending)",
         "assert sort_dictionary_by_value({'a': 3, 'b': 1, 'c': 2}) == [('b', 1), ('c', 2), ('a', 3)]\n"),

        ("sort_dictionary_by_key", "def sort_dictionary_by_key(d: dict[str, Any]) -> list[tuple[str, Any]]:",
         "Returns sorted list of (key, value) pairs ordered alphabetically by key.",
         "Use sorted(d.items(), key=lambda x: x[0]).",
         "    return sorted(d.items(), key=lambda x: x[0])",
         "assert sort_dictionary_by_key({'b': 1, 'a': 2}) == [('a', 2), ('b', 1)]\n"),

        ("sorted_unique_descending", "def sorted_unique_descending(nums: list[int]) -> list[int]:",
         "Returns unique elements sorted in descending order.",
         "Use sorted(set(nums), reverse=True).",
         "    return sorted(set(nums), reverse=True)",
         "assert sorted_unique_descending([1, 3, 2, 3, 1]) == [3, 2, 1]\nassert sorted_unique_descending([]) == []\n"),

        ("min_max_tuple_calc", "def min_max_tuple_calc(nums: list[float]) -> tuple[float, float] | None:",
         "Returns (min(nums), max(nums)) or None if empty.",
         "min and max built-ins.",
         "    if not nums: return None\n    return (min(nums), max(nums))",
         "assert min_max_tuple_calc([3.0, 1.0, 5.0]) == (1.0, 5.0)\nassert min_max_tuple_calc([]) is None\n"),

        ("filter_truthy_dict", "def filter_truthy_dict(d: dict[str, Any]) -> dict[str, Any]:",
         "Filters dictionary keeping only items with truthy values.",
         "Dict comprehension {k: v for k, v in d.items() if v}.",
         "    return {k: v for k, v in d.items() if v}",
         "assert filter_truthy_dict({'a': 1, 'b': 0, 'c': '', 'd': 'yes'}) == {'a': 1, 'd': 'yes'}\n"),

        ("set_symmetric_difference", "def set_symmetric_difference(l1: list[int], l2: list[int]) -> list[int]:",
         "Returns sorted list of elements in either l1 or l2 but not both.",
         "sorted(set(l1) ^ set(l2)).",
         "    return sorted(set(l1) ^ set(l2))",
         "assert set_symmetric_difference([1, 2, 3], [2, 3, 4]) == [1, 4]\nassert set_symmetric_difference([], [1]) == [1]\n"),

        ("set_difference_ordered", "def set_difference_ordered(l1: list[int], l2: list[int]) -> list[int]:",
         "Returns elements in l1 that are not in l2, preserving l1's relative order.",
         "s2 = set(l2); [x for x in l1 if x not in s2].",
         "    s2 = set(l2)\n    return [x for x in l1 if x not in s2]",
         "assert set_difference_ordered([1, 2, 3, 2], [2]) == [1, 3]\n"),

        ("find_all_indices_builtins", "def find_all_indices_builtins(items: list[Any], target: Any) -> list[int]:",
         "Returns all indices where items[i] == target using enumerate.",
         "[i for i, x in enumerate(items) if x == target].",
         "    return [i for i, x in enumerate(items) if x == target]",
         "assert find_all_indices_builtins([1, 2, 3, 2, 2], 2) == [1, 3, 4]\nassert find_all_indices_builtins([], 1) == []\n"),

        ("count_matching_condition", "def count_matching_condition(nums: list[int], divisor: int) -> int:",
         "Counts numbers divisible by divisor using generator expression and sum.",
         "sum(1 for x in nums if x % divisor == 0).",
         "    if divisor == 0: raise ValueError('Divisor cannot be 0')\n    return sum(1 for x in nums if x % divisor == 0)",
         "assert count_matching_condition([10, 15, 22, 30], 5) == 3\n"),

        ("split_list_half_halves", "def split_list_half_halves(items: list[Any]) -> tuple[list[Any], list[Any]]:",
         "Splits list into two halves (first half gets extra element if odd length).",
         "mid = (len(items) + 1) // 2.",
         "    mid = (len(items) + 1) // 2\n    return (items[:mid], items[mid:])",
         "assert split_list_half_halves([1, 2, 3, 4, 5]) == ([1, 2, 3], [4, 5])\nassert split_list_half_halves([]) == ([], [])\n"),

        ("itertools_repeat_values", "def itertools_repeat_values(val: Any, n: int) -> list[Any]:",
         "Repeats value n times using itertools.repeat.",
         "list(itertools.repeat(val, max(0, n))).",
         "    import itertools\n    return list(itertools.repeat(val, max(0, n)))",
         "assert itertools_repeat_values('x', 3) == ['x', 'x', 'x']\nassert itertools_repeat_values('x', 0) == []\n"),

        ("itertools_cycle_take", "def itertools_cycle_take(items: list[Any], n: int) -> list[Any]:",
         "Takes first n items by cycling over items.",
         "Use modulo index or itertools.islice with itertools.cycle.",
         "    if not items or n <= 0: return []\n    return [items[i % len(items)] for i in range(n)]",
         "assert itertools_cycle_take([1, 2], 5) == [1, 2, 1, 2, 1]\nassert itertools_cycle_take([], 3) == []\n"),

        ("itertools_starmap_powers", "def itertools_starmap_powers(pairs: list[tuple[int, int]]) -> list[int]:",
         "Computes [pow(b, e) for b, e in pairs] using itertools.starmap.",
         "list(itertools.starmap(pow, pairs)).",
         "    import itertools\n    return list(itertools.starmap(pow, pairs))",
         "assert itertools_starmap_powers([(2, 3), (3, 2), (5, 0)]) == [8, 9, 1]\n"),

        ("dict_values_sorted_by_len", "def dict_values_sorted_by_len(words_dict: dict[str, str]) -> list[str]:",
         "Returns dictionary values sorted by string length ascending.",
         "sorted(words_dict.values(), key=len).",
         "    return sorted(words_dict.values(), key=len)",
         "assert dict_values_sorted_by_len({'a': 'elephant', 'b': 'cat', 'c': 'doggy'}) == ['cat', 'doggy', 'elephant']\n"),

        ("all_elements_positive_check", "def all_elements_positive_check(nums: list[float]) -> bool:",
         "Checks if all numbers are strictly greater than 0 using built-in all.",
         "all(x > 0 for x in nums).",
         "    return all(x > 0 for x in nums)",
         "assert all_elements_positive_check([1.0, 2.5, 0.1]) is True\nassert all_elements_positive_check([1.0, -0.5]) is False\nassert all_elements_positive_check([]) is True\n"),

        ("any_elements_negative_check", "def any_elements_negative_check(nums: list[float]) -> bool:",
         "Checks if any number is strictly less than 0 using built-in any.",
         "any(x < 0 for x in nums).",
         "    return any(x < 0 for x in nums)",
         "assert any_elements_negative_check([1.0, -2.0, 3.0]) is True\nassert any_elements_negative_check([1.0, 0.0]) is False\nassert any_elements_negative_check([]) is False\n"),

        ("reversed_enumeration_pairs", "def reversed_enumeration_pairs(items: list[Any]) -> list[tuple[int, Any]]:",
         "Returns list of (original_index, item) iterating in reverse order.",
         "list(reversed(list(enumerate(items)))).",
         "    return list(reversed(list(enumerate(items))))",
         "assert reversed_enumeration_pairs(['a', 'b', 'c']) == [(2, 'c'), (1, 'b'), (0, 'a')]\nassert reversed_enumeration_pairs([]) == []\n"),

        ("map_square_integers", "def map_square_integers(nums: list[int]) -> list[int]:",
         "Squares every integer using built-in map.",
         "list(map(lambda x: x * x, nums)).",
         "    return list(map(lambda x: x * x, nums))",
         "assert map_square_integers([1, 2, 3]) == [1, 4, 9]\nassert map_square_integers([]) == []\n"),

        ("filter_even_integers", "def filter_even_integers(nums: list[int]) -> list[int]:",
         "Filters even numbers using built-in filter.",
         "list(filter(lambda x: x % 2 == 0, nums)).",
         "    return list(filter(lambda x: x % 2 == 0, nums))",
         "assert filter_even_integers([1, 2, 3, 4, 5, 6]) == [2, 4, 6]\nassert filter_even_integers([1, 3]) == []\n"),

        ("reduce_sum_integers", "def reduce_sum_integers(nums: list[int], start: int = 0) -> int:",
         "Sums numbers using functools.reduce and operator.add.",
         "functools.reduce(operator.add, nums, start).",
         "    import functools\n    import operator\n    return functools.reduce(operator.add, nums, start)",
         "assert reduce_sum_integers([1, 2, 3, 4]) == 10\nassert reduce_sum_integers([], start=5) == 5\n"),

        ("dict_from_two_lists", "def dict_from_two_lists(keys: list[Any], values: list[Any]) -> dict[Any, Any]:",
         "Constructs dictionary from keys and values lists using zip.",
         "dict(zip(keys, values)).",
         "    return dict(zip(keys, values))",
         "assert dict_from_two_lists(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}\nassert dict_from_two_lists([], []) == {}\n"),

        ("bisect_insert_index", "def bisect_insert_index(sorted_list: list[int], val: int) -> int:",
         "Finds 0-based insertion point to keep list sorted using bisect.bisect_left.",
         "import bisect; bisect.bisect_left(sorted_list, val).",
         "    import bisect\n    return bisect.bisect_left(sorted_list, val)",
         "assert bisect_insert_index([1, 3, 5, 7], 4) == 2\nassert bisect_insert_index([1, 3, 5], 0) == 0\n"),

        ("ordered_dict_lru_step", "def ordered_dict_lru_step(keys: list[str]) -> list[str]:",
         "Simulates accessing keys in OrderedDict with move_to_end, returning final keys.",
         "collections.OrderedDict.move_to_end.",
         "    from collections import OrderedDict\n    od = OrderedDict()\n    for k in keys:\n        if k in od: od.move_to_end(k)\n        od[k] = True\n    return list(od.keys())",
         "assert ordered_dict_lru_step(['a', 'b', 'a', 'c']) == ['b', 'a', 'c']\n"),

        ("counter_subtract_multiset", "def counter_subtract_multiset(s1: list[Any], s2: list[Any]) -> list[Any]:",
         "Subtracts multiset s2 from s1 returning remaining elements sorted.",
         "Counter(s1) - Counter(s2).",
         "    from collections import Counter\n    diff = Counter(s1) - Counter(s2)\n    return sorted(list(diff.elements()))",
         "assert counter_subtract_multiset([1, 1, 2, 3], [1, 2]) == [1, 3]\n"),

        ("itertools_accumulate_max", "def itertools_accumulate_max(nums: list[int]) -> list[int]:",
         "Returns running maximums using itertools.accumulate and max.",
         "list(itertools.accumulate(nums, max)).",
         "    import itertools\n    if not nums: return []\n    return list(itertools.accumulate(nums, max))",
         "assert itertools_accumulate_max([3, 1, 4, 1, 5]) == [3, 3, 4, 4, 5]\nassert itertools_accumulate_max([]) == []\n"),

        ("itertools_accumulate_min", "def itertools_accumulate_min(nums: list[int]) -> list[int]:",
         "Returns running minimums using itertools.accumulate and min.",
         "list(itertools.accumulate(nums, min)).",
         "    import itertools\n    if not nums: return []\n    return list(itertools.accumulate(nums, min))",
         "assert itertools_accumulate_min([3, 1, 4, 0, 5]) == [3, 1, 1, 0, 0]\n"),

        ("string_join_with_prefix_suffix", "def string_join_with_prefix_suffix(items: list[str], prefix: str, suffix: str, sep: str = ', ') -> str:",
         "Joins formatted items f'{prefix}{x}{suffix}' with sep.",
         "sep.join(f'{prefix}{x}{suffix}' for x in items).",
         "    return sep.join(f'{prefix}{x}{suffix}' for x in items)",
         "assert string_join_with_prefix_suffix(['a', 'b'], '[', ']') == '[a], [b]'\nassert string_join_with_prefix_suffix([], '<', '>') == ''\n"),

        ("filter_truthy_values_only", "def filter_truthy_values_only(items: list[Any]) -> list[Any]:",
         "Filters out all falsy values (None, 0, '', False, []) keeping truthy.",
         "[x for x in items if x].",
         "    return [x for x in items if x]",
         "assert filter_truthy_values_only([1, 0, 'a', '', None, True]) == [1, 'a', True]\n"),

        ("count_distinct_types", "def count_distinct_types(items: list[Any]) -> int:",
         "Counts number of distinct Python types among elements in items.",
         "len({type(x) for x in items}).",
         "    return len({type(x) for x in items})",
         "assert count_distinct_types([1, 'a', 2.0, 3, 'b']) == 3\nassert count_distinct_types([]) == 0\n"),

        ("zip_into_dict_reverse", "def zip_into_dict_reverse(keys: list[str], values: list[int]) -> dict[str, int]:",
         "Pairs keys and values reversed using zip(reversed(keys), reversed(values)).",
         "dict(zip(reversed(keys), reversed(values))).",
         "    return dict(zip(reversed(keys), reversed(values)))",
         "assert zip_into_dict_reverse(['a', 'b'], [1, 2]) == {'b': 2, 'a': 1}\n"),

        ("count_elements_in_frequency_range", "def count_elements_in_frequency_range(items: list[Any], min_freq: int, max_freq: int) -> int:",
         "Counts number of distinct elements appearing between min_freq and max_freq times.",
         "Use collections.Counter.",
         "    from collections import Counter\n    counts = Counter(items)\n    return sum(1 for c in counts.values() if min_freq <= c <= max_freq)",
         "assert count_elements_in_frequency_range(['a', 'a', 'b', 'c', 'c', 'c'], 2, 3) == 2\n"),

        ("extract_first_truthy", "def extract_first_truthy(items: list[Any], default: Any = None) -> Any:",
         "Returns first truthy element using next and generator with default.",
         "next((x for x in items if x), default).",
         "    return next((x for x in items if x), default)",
         "assert extract_first_truthy([0, '', 'found', 42]) == 'found'\nassert extract_first_truthy([0, None], default=-1) == -1\n"),

        ("itertools_count_take_n", "def itertools_count_take_n(start: int, step: int, n: int) -> list[int]:",
         "Generates first n values of arithmetic progression starting at start with step.",
         "itertools.islice(itertools.count(start, step), n).",
         "    import itertools\n    return list(itertools.islice(itertools.count(start, step), n))",
         "assert itertools_count_take_n(2, 3, 4) == [2, 5, 8, 11]\nassert itertools_count_take_n(0, 1, 0) == []\n"),

        ("chain_with_separator_items", "def chain_with_separator_items(groups: list[list[Any]], sep: Any = '---') -> list[Any]:",
         "Flattens groups inserting separator sep between consecutive groups.",
         "Loop groups and insert sep.",
         "    if not groups: return []\n    res = list(groups[0])\n    for g in groups[1:]:\n        res.append(sep)\n        res.extend(g)\n    return res",
         "assert chain_with_separator_items([[1, 2], [3, 4]], sep=0) == [1, 2, 0, 3, 4]\nassert chain_with_separator_items([[1]]) == [1]\n"),

        ("filter_by_type_isinstance", "def filter_by_type_isinstance(items: list[Any], target_type: type) -> list[Any]:",
         "Returns elements that are instances of target_type using isinstance.",
         "[x for x in items if isinstance(x, target_type)].",
         "    return [x for x in items if isinstance(x, target_type)]",
         "assert filter_by_type_isinstance([1, 'a', 2, 3.5, 'b'], int) == [1, 2]\nassert filter_by_type_isinstance(['a', 'b'], int) == []\n"),

        ("dict_values_product", "def dict_values_product(d: dict[str, int]) -> int:",
         "Returns product of all integer values in dictionary, or 1 if empty.",
         "functools.reduce with operator.mul.",
         "    import functools\n    import operator\n    if not d: return 1\n    return functools.reduce(operator.mul, d.values(), 1)",
         "assert dict_values_product({'a': 2, 'b': 3, 'c': 4}) == 24\nassert dict_values_product({}) == 1\n"),

        ("sliding_window_average_builtins", "def sliding_window_average_builtins(nums: list[float], k: int) -> list[float]:",
         "Returns list of moving averages of size k.",
         "Slide window of size k.",
         "    if k <= 0 or len(nums) < k: return []\n    curr = sum(nums[:k])\n    res = [curr / k]\n    for i in range(k, len(nums)):\n        curr += nums[i] - nums[i - k]\n        res.append(curr / k)\n    return res",
         "assert sliding_window_average_builtins([1.0, 2.0, 3.0, 4.0], 2) == [1.5, 2.5, 3.5]\nassert sliding_window_average_builtins([1.0], 2) == []\n")
    ]

    for name, sig, desc, hint, sol, test in builtins_specs:
        tasks.append(create_task(
            name=name,
            signature=sig,
            doc_desc=desc,
            doctests=[],
            hint=hint,
            solution=sol,
            test=test
        ))

    return tasks
