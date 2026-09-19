"""
Category 2: Data Structures & Collections (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_data_structures_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Invert Binary Tree
    tasks.append(create_task(
        name="invert_binary_tree_structure",
        signature="def invert_binary_tree_structure(tree: dict | None) -> dict | None:",
        doc_desc="Recursively inverts a binary tree represented as {'val': x, 'left': l, 'right': r}.",
        doctests=[">>> invert_binary_tree_structure({'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}})['left']['val']", "3"],
        hint="If tree is None return None. Otherwise recursively swap inverted right and left subtrees.",
        solution="    if tree is None:\n        return None\n    return {\n        'val': tree['val'],\n        'left': invert_binary_tree_structure(tree.get('right')),\n        'right': invert_binary_tree_structure(tree.get('left'))\n    }",
        test="t = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}\ninv = invert_binary_tree_structure(t)\nassert inv['left']['val'] == 3\nassert inv['right']['val'] == 2\nassert invert_binary_tree_structure(None) is None\n"
    ))

    # 2. Merge Intervals
    tasks.append(create_task(
        name="merge_intervals",
        signature="def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:",
        doc_desc="Merges all overlapping intervals and returns sorted non-overlapping intervals.",
        doctests=[">>> merge_intervals([[1, 3], [2, 6], [8, 10]])", "[[1, 6], [8, 10]]"],
        hint="Sort intervals by start time. Iterate and merge with the last interval if curr[0] <= last[1], otherwise append.",
        solution="    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [list(intervals[0])]\n    for curr in intervals[1:]:\n        prev = merged[-1]\n        if curr[0] <= prev[1]:\n            prev[1] = max(prev[1], curr[1])\n        else:\n            merged.append(list(curr))\n    return merged",
        test="assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]\nassert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]\n"
    ))

    # 3. Is Valid Parentheses
    tasks.append(create_task(
        name="is_valid_parentheses",
        signature="def is_valid_parentheses(s: str) -> bool:",
        doc_desc="Determines if input string of brackets '()', '[]', '{}' is properly paired and closed.",
        doctests=[">>> is_valid_parentheses('()[]{}')", "True", ">>> is_valid_parentheses('(]')", "False"],
        hint="Use a stack. Push opening brackets, and for closing brackets, pop from stack and check matching pair.",
        solution="    stack = []\n    mapping = {')': '(', ']': '[', '}': '{'}\n    for char in s:\n        if char in mapping:\n            top = stack.pop() if stack else '#'\n            if mapping[char] != top:\n                return False\n        elif char in mapping.values():\n            stack.append(char)\n    return not stack",
        test="assert is_valid_parentheses('()[]{}') is True\nassert is_valid_parentheses('(]') is False\nassert is_valid_parentheses('([{}])') is True\nassert is_valid_parentheses('(') is False\nassert is_valid_parentheses('') is True\n"
    ))

    # 4. Two Sum Indices
    tasks.append(create_task(
        name="two_sum_indices",
        signature="def two_sum_indices(nums: list[int], target: int) -> tuple[int, int] | None:",
        doc_desc="Returns 0-based indices of two distinct numbers in nums that add up to target, or None.",
        doctests=[">>> two_sum_indices([2, 7, 11, 15], 9)", "(0, 1)"],
        hint="Use a hash map to store seen numbers and their indices: if target - num in seen, return (seen[target - num], i).",
        solution="    seen = {}\n    for i, n in enumerate(nums):\n        comp = target - n\n        if comp in seen:\n            return (seen[comp], i)\n        seen[n] = i\n    return None",
        test="assert two_sum_indices([2, 7, 11, 15], 9) == (0, 1)\nassert two_sum_indices([3, 2, 4], 6) == (1, 2)\nassert two_sum_indices([1, 2], 10) is None\nassert two_sum_indices([3, 3], 6) == (0, 1)\n"
    ))

    # 5. Evaluate Postfix Notation
    tasks.append(create_task(
        name="evaluate_postfix_notation",
        signature="def evaluate_postfix_notation(tokens: list[str]) -> int:",
        doc_desc="Evaluates Reverse Polish Notation (postfix) arithmetic expressions (+, -, *, //).",
        doctests=[">>> evaluate_postfix_notation(['2', '1', '+', '3', '*'])", "9"],
        hint="Use a stack. Push numbers. When an operator is encountered, pop b, pop a, apply operation (a // b for division), and push result.",
        solution="    stack = []\n    for t in tokens:\n        if t in ('+', '-', '*', '/'):\n            b = stack.pop()\n            a = stack.pop()\n            if t == '+': stack.append(a + b)\n            elif t == '-': stack.append(a - b)\n            elif t == '*': stack.append(a * b)\n            elif t == '/': stack.append(int(a / b))\n        else:\n            stack.append(int(t))\n    return stack[0]",
        test="assert evaluate_postfix_notation(['2', '1', '+', '3', '*']) == 9\nassert evaluate_postfix_notation(['4', '13', '5', '/', '+']) == 6\nassert evaluate_postfix_notation(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22\n"
    ))

    # 6. Binary Tree Max Depth
    tasks.append(create_task(
        name="binary_tree_max_depth",
        signature="def binary_tree_max_depth(root: dict | None) -> int:",
        doc_desc="Calculates maximum depth of a binary tree represented as {'val': v, 'left': l, 'right': r}.",
        doctests=[">>> binary_tree_max_depth({'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': None})", "2"],
        hint="Base case: if root is None return 0. Recursively compute 1 + max(depth(left), depth(right)).",
        solution="    if root is None:\n        return 0\n    return 1 + max(binary_tree_max_depth(root.get('left')), binary_tree_max_depth(root.get('right')))",
        test="assert binary_tree_max_depth({'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': None}) == 2\nassert binary_tree_max_depth(None) == 0\nassert binary_tree_max_depth({'val': 1, 'left': None, 'right': None}) == 1\n"
    ))

    # 7. Binary Tree Is Symmetric
    tasks.append(create_task(
        name="binary_tree_is_symmetric",
        signature="def binary_tree_is_symmetric(root: dict | None) -> bool:",
        doc_desc="Checks if a binary tree is a mirror of itself around its center.",
        doctests=[">>> binary_tree_is_symmetric({'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 2, 'left': None, 'right': None}})", "True"],
        hint="Helper function is_mirror(t1, t2): if both None return True; if one None return False; check vals equal and mirror(t1.left, t2.right) and mirror(t1.right, t2.left).",
        solution="    def is_mirror(t1, t2):\n        if not t1 and not t2:\n            return True\n        if not t1 or not t2:\n            return False\n        return (t1['val'] == t2['val'] and\n                is_mirror(t1.get('left'), t2.get('right')) and\n                is_mirror(t1.get('right'), t2.get('left')))\n    if not root:\n        return True\n    return is_mirror(root.get('left'), root.get('right'))",
        test="t1 = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 2, 'left': None, 'right': None}}\nassert binary_tree_is_symmetric(t1) is True\nt2 = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}\nassert binary_tree_is_symmetric(t2) is False\nassert binary_tree_is_symmetric(None) is True\n"
    ))

    # 8. Lowest Common Ancestor BST
    tasks.append(create_task(
        name="lowest_common_ancestor_bst",
        signature="def lowest_common_ancestor_bst(root: dict | None, p: int, q: int) -> int | None:",
        doc_desc="Finds lowest common ancestor value of two node values p and q in a Binary Search Tree.",
        doctests=[">>> lowest_common_ancestor_bst({'val': 6, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 8, 'left': None, 'right': None}}, 2, 8)", "6"],
        hint="Traverse from root: if both p and q < root['val'], go left; if both > root['val'], go right; otherwise root is split point.",
        solution="    curr = root\n    while curr:\n        val = curr['val']\n        if p < val and q < val:\n            curr = curr.get('left')\n        elif p > val and q > val:\n            curr = curr.get('right')\n        else:\n            return val\n    return None",
        test="tree = {'val': 6, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 8, 'left': None, 'right': None}}\nassert lowest_common_ancestor_bst(tree, 2, 8) == 6\nassert lowest_common_ancestor_bst(tree, 2, 2) == 2\nassert lowest_common_ancestor_bst(None, 1, 2) is None\n"
    ))

    # 9. Level Order Traversal
    tasks.append(create_task(
        name="binary_tree_level_order",
        signature="def binary_tree_level_order(root: dict | None) -> list[list[int]]:",
        doc_desc="Returns the level-order traversal of a binary tree's node values from top to bottom, left to right.",
        doctests=[">>> binary_tree_level_order({'val': 3, 'left': {'val': 9, 'left': None, 'right': None}, 'right': {'val': 20, 'left': None, 'right': None}})", "[[3], [9, 20]]"],
        hint="Use collections.deque for BFS. Loop level by level, popping len(queue) nodes and collecting their values and children.",
        solution="    if not root:\n        return []\n    from collections import deque\n    res = []\n    q = deque([root])\n    while q:\n        level = []\n        for _ in range(len(q)):\n            node = q.popleft()\n            level.append(node['val'])\n            if node.get('left'):\n                q.append(node['left'])\n            if node.get('right'):\n                q.append(node['right'])\n        res.append(level)\n    return res",
        test="t = {'val': 3, 'left': {'val': 9, 'left': None, 'right': None}, 'right': {'val': 20, 'left': None, 'right': None}}\nassert binary_tree_level_order(t) == [[3], [9, 20]]\nassert binary_tree_level_order(None) == []\n"
    ))

    # 10. Directed Graph Has Cycle
    tasks.append(create_task(
        name="graph_has_cycle_directed",
        signature="def graph_has_cycle_directed(n: int, edges: list[tuple[int, int]]) -> bool:",
        doc_desc="Determines if a directed graph with n vertices (0..n-1) contains any directed cycles.",
        doctests=[">>> graph_has_cycle_directed(2, [(0, 1), (1, 0)])", "True", ">>> graph_has_cycle_directed(2, [(0, 1)])", "False"],
        hint="Use 3-color DFS: 0=unvisited, 1=visiting, 2=visited. If an edge leads to a node currently 'visiting', a cycle exists.",
        solution="    adj = [[] for _ in range(n)]\n    for u, v in edges:\n        adj[u].append(v)\n    visited = [0] * n  # 0: unvisited, 1: visiting, 2: visited\n    def dfs(u):\n        visited[u] = 1\n        for v in adj[u]:\n            if visited[v] == 1:\n                return True\n            if visited[v] == 0 and dfs(v):\n                return True\n        visited[u] = 2\n        return False\n    for i in range(n):\n        if visited[i] == 0:\n            if dfs(i):\n                return True\n    return False",
        test="assert graph_has_cycle_directed(2, [(0, 1), (1, 0)]) is True\nassert graph_has_cycle_directed(2, [(0, 1)]) is False\nassert graph_has_cycle_directed(3, [(0, 1), (1, 2), (2, 0)]) is True\nassert graph_has_cycle_directed(3, [(0, 1), (1, 2)]) is False\n"
    ))

    # 11. Graph Path Exists
    tasks.append(create_task(
        name="graph_path_exists",
        signature="def graph_path_exists(n: int, edges: list[tuple[int, int]], source: int, dest: int) -> bool:",
        doc_desc="Checks if there is a valid path between source and destination in an undirected graph.",
        doctests=[">>> graph_path_exists(3, [(0, 1), (1, 2)], 0, 2)", "True"],
        hint="Build adjacency list for undirected graph. Run BFS from source with a visited set and check if dest is reached.",
        solution="    if source == dest:\n        return True\n    from collections import deque\n    adj = [[] for _ in range(n)]\n    for u, v in edges:\n        adj[u].append(v)\n        adj[v].append(u)\n    visited = {source}\n    q = deque([source])\n    while q:\n        curr = q.popleft()\n        if curr == dest:\n            return True\n        for nxt in adj[curr]:\n            if nxt not in visited:\n                visited.add(nxt)\n                q.append(nxt)\n    return False",
        test="assert graph_path_exists(3, [(0, 1), (1, 2)], 0, 2) is True\nassert graph_path_exists(3, [(0, 1)], 0, 2) is False\nassert graph_path_exists(1, [], 0, 0) is True\n"
    ))

    # 12. Topological Sort (Kahn's)
    tasks.append(create_task(
        name="topological_sort_kahn",
        signature="def topological_sort_kahn(n: int, edges: list[tuple[int, int]]) -> list[int] | None:",
        doc_desc="Returns a topological ordering of DAG with n nodes, or None if the graph has a cycle.",
        doctests=[">>> topological_sort_kahn(2, [(0, 1)])", "[0, 1]"],
        hint="Calculate in-degrees. Seed queue with nodes of in-degree 0. As nodes are removed, decrement neighbors' in-degrees.",
        solution="    from collections import deque\n    adj = [[] for _ in range(n)]\n    in_deg = [0] * n\n    for u, v in edges:\n        adj[u].append(v)\n        in_deg[v] += 1\n    q = deque([i for i in range(n) if in_deg[i] == 0])\n    order = []\n    while q:\n        u = q.popleft()\n        order.append(u)\n        for v in adj[u]:\n            in_deg[v] -= 1\n            if in_deg[v] == 0:\n                q.append(v)\n    return order if len(order) == n else None",
        test="assert topological_sort_kahn(2, [(0, 1)]) == [0, 1]\nassert topological_sort_kahn(2, [(0, 1), (1, 0)]) is None\nassert len(topological_sort_kahn(3, [(0, 1), (0, 2)])) == 3\n"
    ))

    # 13. Count Connected Components
    tasks.append(create_task(
        name="count_connected_components",
        signature="def count_connected_components(n: int, edges: list[tuple[int, int]]) -> int:",
        doc_desc="Counts the number of connected components in an undirected graph with n nodes.",
        doctests=[">>> count_connected_components(5, [(0, 1), (1, 2), (3, 4)])", "2"],
        hint="Build adjacency graph. Iterate nodes 0..n-1: if node not in visited, increment count and run BFS/DFS to mark component.",
        solution="    adj = [[] for _ in range(n)]\n    for u, v in edges:\n        adj[u].append(v)\n        adj[v].append(u)\n    visited = set()\n    comps = 0\n    for i in range(n):\n        if i not in visited:\n            comps += 1\n            stack = [i]\n            visited.add(i)\n            while stack:\n                u = stack.pop()\n                for v in adj[u]:\n                    if v not in visited:\n                        visited.add(v)\n                        stack.append(v)\n    return comps",
        test="assert count_connected_components(5, [(0, 1), (1, 2), (3, 4)]) == 2\nassert count_connected_components(5, [(0, 1), (1, 2), (2, 3), (3, 4)]) == 1\nassert count_connected_components(3, []) == 3\n"
    ))

    # 14. Dijkstra Shortest Paths
    tasks.append(create_task(
        name="dijkstra_shortest_paths",
        signature="def dijkstra_shortest_paths(n: int, edges: list[tuple[int, int, float]], start: int) -> dict[int, float]:",
        doc_desc="Computes shortest path distances from start node to all reachable nodes in directed weighted graph.",
        doctests=[">>> dijkstra_shortest_paths(3, [(0, 1, 4.0), (1, 2, 1.0), (0, 2, 6.0)], 0)", "{0: 0.0, 1: 4.0, 2: 5.0}"],
        hint="Use heapq priority queue with (dist, node). Track dists dict; skip stale distances.",
        solution="    import heapq\n    adj = [[] for _ in range(n)]\n    for u, v, w in edges:\n        adj[u].append((v, w))\n    dists = {start: 0.0}\n    pq = [(0.0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dists[u]:\n            continue\n        for v, w in adj[u]:\n            if v not in dists or d + w < dists[v]:\n                dists[v] = d + w\n                heapq.heappush(pq, (d + w, v))\n    return dists",
        test="d = dijkstra_shortest_paths(3, [(0, 1, 4.0), (1, 2, 1.0), (0, 2, 6.0)], 0)\nassert d[0] == 0.0 and d[1] == 4.0 and d[2] == 5.0\nassert dijkstra_shortest_paths(2, [], 0) == {0: 0.0}\n"
    ))

    # 15. Min Stack Class
    tasks.append(create_task(
        name="MinStack",
        signature="class MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []",
        doc_desc="Stack data structure supporting push, pop, top, and get_min in O(1) time.",
        doctests=[">>> s = MinStack(); s.push(3); s.push(1); s.get_min()", "1"],
        hint="Maintain a secondary stack min_stack where each entry is min(val, min_stack[-1]) to track running minimums.",
        solution="    def push(self, val: int) -> None:\n        self.stack.append(val)\n        m = val if not self.min_stack else min(val, self.min_stack[-1])\n        self.min_stack.append(m)\n    def pop(self) -> int | None:\n        if not self.stack: return None\n        self.min_stack.pop()\n        return self.stack.pop()\n    def top(self) -> int | None:\n        return self.stack[-1] if self.stack else None\n    def get_min(self) -> int | None:\n        return self.min_stack[-1] if self.min_stack else None",
        test="s = MinStack()\ns.push(3)\ns.push(5)\nassert s.get_min() == 3\ns.push(2)\nassert s.get_min() == 2\nassert s.pop() == 2\nassert s.get_min() == 3\nassert s.top() == 5\n"
    ))

    # 16. Queue Using Two Stacks
    tasks.append(create_task(
        name="QueueTwoStacks",
        signature="class QueueTwoStacks:\n    def __init__(self):\n        self.in_stack = []\n        self.out_stack = []",
        doc_desc="FIFO queue implemented using two LIFO stacks with amortized O(1) enqueue and dequeue.",
        doctests=[">>> q = QueueTwoStacks(); q.enqueue(1); q.enqueue(2); q.dequeue()", "1"],
        hint="Push new elements to in_stack. For dequeue/peek: if out_stack is empty, pop all from in_stack and push to out_stack.",
        solution="    def enqueue(self, val: Any) -> None:\n        self.in_stack.append(val)\n    def _shift(self) -> None:\n        if not self.out_stack:\n            while self.in_stack:\n                self.out_stack.append(self.in_stack.pop())\n    def dequeue(self) -> Any | None:\n        self._shift()\n        return self.out_stack.pop() if self.out_stack else None\n    def peek(self) -> Any | None:\n        self._shift()\n        return self.out_stack[-1] if self.out_stack else None\n    def is_empty(self) -> bool:\n        return not self.in_stack and not self.out_stack",
        test="q = QueueTwoStacks()\nq.enqueue(1)\nq.enqueue(2)\nassert q.dequeue() == 1\nq.enqueue(3)\nassert q.peek() == 2\nassert q.dequeue() == 2\nassert q.dequeue() == 3\nassert q.is_empty() is True\n"
    ))

    # 17. Circular Buffer Queue
    tasks.append(create_task(
        name="CircularBuffer",
        signature="class CircularBuffer:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.buf = [None] * capacity\n        self.head = 0\n        self.tail = 0\n        self.size = 0",
        doc_desc="Fixed-size circular ring buffer queue supporting enqueue, dequeue, is_full, and is_empty.",
        doctests=[">>> cb = CircularBuffer(2); cb.enqueue('a'); cb.enqueue('b'); cb.is_full()", "True"],
        hint="Track head and tail pointers modulo capacity, along with current size. Enqueue at tail, dequeue at head.",
        solution="    def enqueue(self, val: Any) -> bool:\n        if self.size == self.cap: return False\n        self.buf[self.tail] = val\n        self.tail = (self.tail + 1) % self.cap\n        self.size += 1\n        return True\n    def dequeue(self) -> Any | None:\n        if self.size == 0: return None\n        val = self.buf[self.head]\n        self.buf[self.head] = None\n        self.head = (self.head + 1) % self.cap\n        self.size -= 1\n        return val\n    def is_full(self) -> bool: return self.size == self.cap\n    def is_empty(self) -> bool: return self.size == 0",
        test="cb = CircularBuffer(2)\nassert cb.enqueue(1) is True\nassert cb.enqueue(2) is True\nassert cb.enqueue(3) is False\nassert cb.is_full() is True\nassert cb.dequeue() == 1\nassert cb.enqueue(4) is True\nassert cb.dequeue() == 2\nassert cb.dequeue() == 4\nassert cb.is_empty() is True\n"
    ))

    # 18. Monotonic Stack Next Greater Element
    tasks.append(create_task(
        name="next_greater_elements",
        signature="def next_greater_elements(nums: list[int]) -> list[int]:",
        doc_desc="For each element, finds the first greater element to its right, or -1 if none exists.",
        doctests=[">>> next_greater_elements([2, 1, 2, 4, 3])", "[4, 2, 4, -1, -1]"],
        hint="Traverse from right to left using a monotonic stack. Pop elements <= curr; top of stack is next greater.",
        solution="    res = [-1] * len(nums)\n    stack = []\n    for i in range(len(nums) - 1, -1, -1):\n        while stack and stack[-1] <= nums[i]:\n            stack.pop()\n        if stack:\n            res[i] = stack[-1]\n        stack.append(nums[i])\n    return res",
        test="assert next_greater_elements([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]\nassert next_greater_elements([1, 2, 3]) == [2, 3, -1]\nassert next_greater_elements([3, 2, 1]) == [-1, -1, -1]\nassert next_greater_elements([]) == []\n"
    ))

    # 19. Sliding Window Maximum
    tasks.append(create_task(
        name="sliding_window_maximum",
        signature="def sliding_window_maximum(nums: list[int], k: int) -> list[int]:",
        doc_desc="Returns the maximum value in every sliding window of size k moving across nums from left to right.",
        doctests=[">>> sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3)", "[3, 3, 5, 5, 6, 7]"],
        hint="Use a deque storing indices of decreasing elements. Remove indices out of window, pop smaller elements from back.",
        solution="    from collections import deque\n    if not nums or k <= 0:\n        return []\n    q = deque()\n    res = []\n    for i, n in enumerate(nums):\n        while q and q[0] < i - k + 1:\n            q.popleft()\n        while q and nums[q[-1]] < n:\n            q.pop()\n        q.append(i)\n        if i >= k - 1:\n            res.append(nums[q[0]])\n    return res",
        test="assert sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]\nassert sliding_window_maximum([1], 1) == [1]\nassert sliding_window_maximum([], 3) == []\n"
    ))

    # 20. Merge Two Sorted Lists
    tasks.append(create_task(
        name="merge_two_sorted_lists",
        signature="def merge_two_sorted_lists(l1: list[int], l2: list[int]) -> list[int]:",
        doc_desc="Merges two sorted lists of integers into a single sorted list in O(n + m) time.",
        doctests=[">>> merge_two_sorted_lists([1, 2, 4], [1, 3, 4])", "[1, 1, 2, 3, 4, 4]"],
        hint="Use two pointers i and j starting at 0. Append smaller element; flush remaining elements when one list finishes.",
        solution="    i, j = 0, 0\n    res = []\n    while i < len(l1) and j < len(l2):\n        if l1[i] <= l2[j]:\n            res.append(l1[i])\n            i += 1\n        else:\n            res.append(l2[j])\n            j += 1\n    res.extend(l1[i:])\n    res.extend(l2[j:])\n    return res",
        test="assert merge_two_sorted_lists([1, 2, 4], [1, 3, 4]) == [1, 1, 2, 3, 4, 4]\nassert merge_two_sorted_lists([], [0]) == [0]\nassert merge_two_sorted_lists([], []) == []\n"
    ))

    # 21. Merge K Sorted Lists
    tasks.append(create_task(
        name="merge_k_sorted_lists",
        signature="def merge_k_sorted_lists(lists: list[list[int]]) -> list[int]:",
        doc_desc="Merges k sorted lists of integers into one sorted list using a min-heap.",
        doctests=[">>> merge_k_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]])", "[1, 1, 2, 3, 4, 4, 5, 6]"],
        hint="Push (lst[0], list_idx, 0) to a min-heap for each non-empty list. Pop min, append to result, and push next element.",
        solution="    import heapq\n    heap = []\n    for idx, lst in enumerate(lists):\n        if lst:\n            heapq.heappush(heap, (lst[0], idx, 0))\n    res = []\n    while heap:\n        val, l_idx, elem_idx = heapq.heappop(heap)\n        res.append(val)\n        if elem_idx + 1 < len(lists[l_idx]):\n            nxt_val = lists[l_idx][elem_idx + 1]\n            heapq.heappush(heap, (nxt_val, l_idx, elem_idx + 1))\n    return res",
        test="assert merge_k_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]]) == [1, 1, 2, 3, 4, 4, 5, 6]\nassert merge_k_sorted_lists([]) == []\nassert merge_k_sorted_lists([[]]) == []\n"
    ))

    # 22. Group Anagrams
    tasks.append(create_task(
        name="group_anagrams",
        signature="def group_anagrams(words: list[str]) -> list[list[str]]:",
        doc_desc="Groups strings that are anagrams of each other together in lists.",
        doctests=[">>> len(group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']))", "3"],
        hint="Use collections.defaultdict(list) keyed by sorted string tuple(sorted(w)). Return list of grouped values.",
        solution="    from collections import defaultdict\n    groups = defaultdict(list)\n    for w in words:\n        groups[tuple(sorted(w))].append(w)\n    return list(groups.values())",
        test="res = group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])\nassert len(res) == 3\nassert sorted([sorted(g) for g in res]) == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]\nassert group_anagrams(['']) == [['']]\n"
    ))

    # 23. Longest Consecutive Sequence
    tasks.append(create_task(
        name="longest_consecutive_sequence",
        signature="def longest_consecutive_sequence(nums: list[int]) -> int:",
        doc_desc="Finds the length of the longest consecutive elements sequence in O(n) time.",
        doctests=[">>> longest_consecutive_sequence([100, 4, 200, 1, 3, 2])", "4"],
        hint="Store elements in a set. For each n in set, if n - 1 not in set (streak start), count while n + length in set.",
        solution="    num_set = set(nums)\n    max_streak = 0\n    for n in num_set:\n        if n - 1 not in num_set:\n            curr = n\n            streak = 1\n            while curr + 1 in num_set:\n                curr += 1\n                streak += 1\n            max_streak = max(max_streak, streak)\n    return max_streak",
        test="assert longest_consecutive_sequence([100, 4, 200, 1, 3, 2]) == 4\nassert longest_consecutive_sequence([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9\nassert longest_consecutive_sequence([]) == 0\n"
    ))

    # 24. Subarray Sum Equals K
    tasks.append(create_task(
        name="subarray_sum_equals_k",
        signature="def subarray_sum_equals_k(nums: list[int], k: int) -> int:",
        doc_desc="Finds total number of contiguous subarrays whose sum equals k in O(n) time.",
        doctests=[">>> subarray_sum_equals_k([1, 1, 1], 2)", "2", ">>> subarray_sum_equals_k([1, 2, 3], 3)", "2"],
        hint="Use a prefix sum frequency dictionary: count += seen_prefix.get(curr_sum - k, 0), then seen_prefix[curr_sum] += 1.",
        solution="    count = 0\n    curr_sum = 0\n    prefix_counts = {0: 1}\n    for n in nums:\n        curr_sum += n\n        count += prefix_counts.get(curr_sum - k, 0)\n        prefix_counts[curr_sum] = prefix_counts.get(curr_sum, 0) + 1\n    return count",
        test="assert subarray_sum_equals_k([1, 1, 1], 2) == 2\nassert subarray_sum_equals_k([1, 2, 3], 3) == 2\nassert subarray_sum_equals_k([1, -1, 0], 0) == 3\nassert subarray_sum_equals_k([], 5) == 0\n"
    ))

    # 25. First Unique Character
    tasks.append(create_task(
        name="first_unique_character",
        signature="def first_unique_character(s: str) -> int:",
        doc_desc="Returns the 0-based index of the first non-repeating character in string s, or -1 if none.",
        doctests=[">>> first_unique_character('leetcode')", "0", ">>> first_unique_character('loveleetcode')", "2"],
        hint="Count character occurrences with collections.Counter(s). Iterate enumerate(s) and return index of first with count == 1.",
        solution="    from collections import Counter\n    counts = Counter(s)\n    for i, ch in enumerate(s):\n        if counts[ch] == 1:\n            return i\n    return -1",
        test="assert first_unique_character('leetcode') == 0\nassert first_unique_character('loveleetcode') == 2\nassert first_unique_character('aabb') == -1\nassert first_unique_character('') == -1\n"
    ))

    # 26. Isomorphic Strings
    tasks.append(create_task(
        name="are_isomorphic_strings",
        signature="def are_isomorphic_strings(s: str, t: str) -> bool:",
        doc_desc="Determines if characters in string s can be uniquely replaced to get string t.",
        doctests=[">>> are_isomorphic_strings('egg', 'add')", "True", ">>> are_isomorphic_strings('foo', 'bar')", "False"],
        hint="Check len(s) == len(t), len(set(s)) == len(set(t)) == len(set(zip(s, t))).",
        solution="    if len(s) != len(t):\n        return False\n    return len(set(s)) == len(set(t)) == len(set(zip(s, t)))",
        test="assert are_isomorphic_strings('egg', 'add') is True\nassert are_isomorphic_strings('foo', 'bar') is False\nassert are_isomorphic_strings('paper', 'title') is True\nassert are_isomorphic_strings('ab', 'aa') is False\n"
    ))

    # 27. Word Pattern Match
    tasks.append(create_task(
        name="word_pattern_match",
        signature="def word_pattern_match(pattern: str, s: str) -> bool:",
        doc_desc="Checks if string s follows the bijection pattern where each letter corresponds to a word.",
        doctests=[">>> word_pattern_match('abba', 'dog cat cat dog')", "True", ">>> word_pattern_match('abba', 'dog cat cat fish')", "False"],
        hint="Split s by whitespace. If len(words) != len(pattern) return False. Check len(set(pattern)) == len(set(words)) == len(set(zip(pattern, words))).",
        solution="    words = s.split()\n    if len(words) != len(pattern):\n        return False\n    return len(set(pattern)) == len(set(words)) == len(set(zip(pattern, words)))",
        test="assert word_pattern_match('abba', 'dog cat cat dog') is True\nassert word_pattern_match('abba', 'dog cat cat fish') is False\nassert word_pattern_match('aaaa', 'dog cat cat dog') is False\nassert word_pattern_match('abba', 'dog dog dog dog') is False\n"
    ))

    # 28. Find All Duplicates
    tasks.append(create_task(
        name="find_all_duplicates_list",
        signature="def find_all_duplicates_list(nums: list[int]) -> list[int]:",
        doc_desc="Returns a list of all elements that appear more than once, sorted in ascending order.",
        doctests=[">>> find_all_duplicates_list([4, 3, 2, 7, 8, 2, 3, 1])", "[2, 3]"],
        hint="Use collections.Counter to count occurrences; filter elements with count > 1 and return sorted.",
        solution="    from collections import Counter\n    counts = Counter(nums)\n    return sorted([k for k, v in counts.items() if v > 1])",
        test="assert find_all_duplicates_list([4, 3, 2, 7, 8, 2, 3, 1]) == [2, 3]\nassert find_all_duplicates_list([1, 1, 2]) == [1]\nassert find_all_duplicates_list([1]) == []\n"
    ))

    # 29. Intersection of Two Arrays
    tasks.append(create_task(
        name="intersection_two_arrays",
        signature="def intersection_two_arrays(nums1: list[int], nums2: list[int]) -> list[int]:",
        doc_desc="Returns unique elements present in both arrays, sorted in ascending order.",
        doctests=[">>> intersection_two_arrays([1, 2, 2, 1], [2, 2])", "[2]"],
        hint="Compute sorted(list(set(nums1) & set(nums2))).",
        solution="    return sorted(list(set(nums1) & set(nums2)))",
        test="assert intersection_two_arrays([1, 2, 2, 1], [2, 2]) == [2]\nassert intersection_two_arrays([4, 9, 5], [9, 4, 9, 8, 4]) == [4, 9]\nassert intersection_two_arrays([], [1, 2]) == []\n"
    ))

    # 30. Union of Sorted Arrays
    tasks.append(create_task(
        name="union_sorted_arrays",
        signature="def union_sorted_arrays(a: list[int], b: list[int]) -> list[int]:",
        doc_desc="Returns sorted union of two lists without duplicates in O(n + m) time.",
        doctests=[">>> union_sorted_arrays([1, 2, 4], [2, 3, 5])", "[1, 2, 3, 4, 5]"],
        hint="Use two pointers on sorted arrays (or convert to sets and sort).",
        solution="    return sorted(list(set(a) | set(b)))",
        test="assert union_sorted_arrays([1, 2, 4], [2, 3, 5]) == [1, 2, 3, 4, 5]\nassert union_sorted_arrays([], [1, 2]) == [1, 2]\nassert union_sorted_arrays([1, 1], [1, 1]) == [1]\n"
    ))

    # 31. Singly Linked List Reverse (Dict representation)
    tasks.append(create_task(
        name="reverse_linked_list_dict",
        signature="def reverse_linked_list_dict(head: dict | None) -> dict | None:",
        doc_desc="Reverses a linked list where nodes are {'val': v, 'next': node}.",
        doctests=[">>> reverse_linked_list_dict({'val': 1, 'next': {'val': 2, 'next': None}})['val']", "2"],
        hint="Iterative pointer reversal: prev = None, curr = head; while curr: nxt = curr['next']; curr['next'] = prev; prev = curr; curr = nxt.",
        solution="    prev = None\n    curr = head\n    while curr:\n        nxt = curr.get('next')\n        curr['next'] = prev\n        prev = curr\n        curr = nxt\n    return prev",
        test="node2 = {'val': 2, 'next': None}\nnode1 = {'val': 1, 'next': node2}\nrev = reverse_linked_list_dict(node1)\nassert rev['val'] == 2 and rev['next']['val'] == 1 and rev['next']['next'] is None\nassert reverse_linked_list_dict(None) is None\n"
    ))

    # 32. Flatten Nested Dictionary
    tasks.append(create_task(
        name="flatten_nested_dict",
        signature="def flatten_nested_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:",
        doc_desc="Flattens a deeply nested dictionary into a single-level dictionary with compound keys.",
        doctests=[">>> flatten_nested_dict({'a': 1, 'b': {'c': 2}})", "{'a': 1, 'b.c': 2}"],
        hint="Recursively traverse: for k, v in d.items(), new_key = f'{parent_key}{sep}{k}' if parent_key else k; recurse if v is dict.",
        solution="    items = {}\n    for k, v in d.items():\n        new_key = f'{parent_key}{sep}{k}' if parent_key else str(k)\n        if isinstance(v, dict):\n            items.update(flatten_nested_dict(v, new_key, sep=sep))\n        else:\n            items[new_key] = v\n    return items",
        test="assert flatten_nested_dict({'a': 1, 'b': {'c': 2}}) == {'a': 1, 'b.c': 2}\nassert flatten_nested_dict({'x': {'y': {'z': 3}}}) == {'x.y.z': 3}\nassert flatten_nested_dict({}) == {}\n"
    ))

    # 33. Trie Prefix Search
    tasks.append(create_task(
        name="Trie",
        signature="class Trie:\n    def __init__(self):\n        self.root = {}",
        doc_desc="Trie (Prefix Tree) supporting insert(word), search(word), and starts_with(prefix).",
        doctests=[">>> t = Trie(); t.insert('apple'); t.search('apple')", "True"],
        hint="Use nested dictionaries where each node is a dict and '#' or '$' marks the end of a word.",
        solution="    def insert(self, word: str) -> None:\n        curr = self.root\n        for ch in word:\n            curr = curr.setdefault(ch, {})\n        curr['#'] = True\n    def search(self, word: str) -> bool:\n        curr = self.root\n        for ch in word:\n            if ch not in curr: return False\n            curr = curr[ch]\n        return '#' in curr\n    def starts_with(self, prefix: str) -> bool:\n        curr = self.root\n        for ch in prefix:\n            if ch not in curr: return False\n            curr = curr[ch]\n        return True",
        test="t = Trie()\nt.insert('apple')\nassert t.search('apple') is True\nassert t.search('app') is False\nassert t.starts_with('app') is True\nt.insert('app')\nassert t.search('app') is True\n"
    ))

    # 34. LRU Cache Simulation
    tasks.append(create_task(
        name="LRUCache",
        signature="class LRUCache:\n    def __init__(self, capacity: int):\n        from collections import OrderedDict\n        self.cap = capacity\n        self.cache = OrderedDict()",
        doc_desc="Least Recently Used (LRU) Cache supporting get(key) and put(key, value) in O(1) time.",
        doctests=[">>> lru = LRUCache(2); lru.put(1, 1); lru.put(2, 2); lru.get(1)", "1"],
        hint="Use collections.OrderedDict. When accessed or updated, use move_to_end(key). If len exceeds capacity, popitem(last=False).",
        solution="    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)",
        test="lru = LRUCache(2)\nlru.put(1, 1)\nlru.put(2, 2)\nassert lru.get(1) == 1\nlru.put(3, 3)\nassert lru.get(2) == -1\nassert lru.get(3) == 3\nassert lru.get(1) == 1\n"
    ))

    # 35. Priority Queue Min-Heap
    tasks.append(create_task(
        name="PriorityQueueMinHeap",
        signature="class PriorityQueueMinHeap:\n    def __init__(self):\n        self.heap = []",
        doc_desc="Min-priority queue storing items with priority weights, supporting push, pop_min, and peek.",
        doctests=[">>> pq = PriorityQueueMinHeap(); pq.push('task1', 5); pq.push('task2', 1); pq.pop_min()", "'task2'"],
        hint="Use heapq module storing tuples (priority, item). Pop returns item of minimum priority.",
        solution="    def push(self, item: Any, priority: float) -> None:\n        import heapq\n        heapq.heappush(self.heap, (priority, item))\n    def pop_min(self) -> Any | None:\n        import heapq\n        return heapq.heappop(self.heap)[1] if self.heap else None\n    def peek(self) -> Any | None:\n        return self.heap[0][1] if self.heap else None\n    def size(self) -> int:\n        return len(self.heap)",
        test="pq = PriorityQueueMinHeap()\npq.push('b', 2)\npq.push('a', 1)\npq.push('c', 3)\nassert pq.pop_min() == 'a'\nassert pq.peek() == 'b'\nassert pq.size() == 2\n"
    ))

    # 36. Sparse Vector Dot Product
    tasks.append(create_task(
        name="sparse_vector_dot_product",
        signature="def sparse_vector_dot_product(vec1: dict[int, float], vec2: dict[int, float]) -> float:",
        doc_desc="Calculates dot product of two sparse vectors represented as {index: non_zero_value}.",
        doctests=[">>> sparse_vector_dot_product({0: 1.0, 3: 2.0}, {0: 4.0, 1: 5.0})", "4.0"],
        hint="Iterate over the smaller dictionary: if key in larger, sum k * v.",
        solution="    if len(vec1) > len(vec2):\n        vec1, vec2 = vec2, vec1\n    total = 0.0\n    for idx, val in vec1.items():\n        if idx in vec2:\n            total += val * vec2[idx]\n    return total",
        test="assert sparse_vector_dot_product({0: 1.0, 3: 2.0}, {0: 4.0, 1: 5.0}) == 4.0\nassert sparse_vector_dot_product({}, {1: 2.0}) == 0.0\nassert sparse_vector_dot_product({1: 3.0}, {1: 2.0}) == 6.0\n"
    ))

    # 37. Disjoint Set Union-Find
    tasks.append(create_task(
        name="DisjointSetUnion",
        signature="class DisjointSetUnion:\n    def __init__(self, n: int):\n        self.parent = list(range(n))\n        self.rank = [0] * n",
        doc_desc="Disjoint Set Union (DSU) data structure with path compression and union by rank.",
        doctests=[">>> dsu = DisjointSetUnion(5); dsu.union(0, 1); dsu.connected(0, 1)", "True"],
        hint="find(x): if parent[x] != x: parent[x] = find(parent[x]); return parent[x]. union(x, y) attaches smaller rank root under larger.",
        solution="    def find(self, x: int) -> int:\n        if self.parent[x] != x:\n            self.parent[x] = self.find(self.parent[x])\n        return self.parent[x]\n    def union(self, x: int, y: int) -> bool:\n        root_x, root_y = self.find(x), self.find(y)\n        if root_x == root_y:\n            return False\n        if self.rank[root_x] < self.rank[root_y]:\n            self.parent[root_x] = root_y\n        elif self.rank[root_x] > self.rank[root_y]:\n            self.parent[root_y] = root_x\n        else:\n            self.parent[root_y] = root_x\n            self.rank[root_x] += 1\n        return True\n    def connected(self, x: int, y: int) -> bool:\n        return self.find(x) == self.find(y)",
        test="dsu = DisjointSetUnion(5)\nassert dsu.connected(0, 1) is False\nassert dsu.union(0, 1) is True\nassert dsu.connected(0, 1) is True\ndsu.union(1, 2)\nassert dsu.connected(0, 2) is True\nassert dsu.connected(0, 3) is False\n"
    ))

    # 38. Binary Search Insert Position
    tasks.append(create_task(
        name="search_insert_position",
        signature="def search_insert_position(nums: list[int], target: int) -> int:",
        doc_desc="Returns the index if target is found in sorted array nums, or the index where it would be inserted.",
        doctests=[">>> search_insert_position([1, 3, 5, 6], 5)", "2", ">>> search_insert_position([1, 3, 5, 6], 2)", "1"],
        hint="Binary search: while low <= high: mid = (low + high) // 2. If nums[mid] == target return mid; adjust low or high.",
        solution="    low, high = 0, len(nums) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if nums[mid] == target:\n            return mid\n        elif nums[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return low",
        test="assert search_insert_position([1, 3, 5, 6], 5) == 2\nassert search_insert_position([1, 3, 5, 6], 2) == 1\nassert search_insert_position([1, 3, 5, 6], 7) == 4\nassert search_insert_position([1, 3, 5, 6], 0) == 0\n"
    ))

    # 39. Find Kth Largest Element
    tasks.append(create_task(
        name="find_kth_largest_element",
        signature="def find_kth_largest_element(nums: list[int], k: int) -> int:",
        doc_desc="Finds the kth largest element in an unsorted array.",
        doctests=[">>> find_kth_largest_element([3, 2, 1, 5, 6, 4], 2)", "5"],
        hint="Maintain a min-heap of size k using heapq: for each num, push and if len > k, pop. Return heap[0].",
        solution="    import heapq\n    heap = []\n    for n in nums:\n        heapq.heappush(heap, n)\n        if len(heap) > k:\n            heapq.heappop(heap)\n    return heap[0]",
        test="assert find_kth_largest_element([3, 2, 1, 5, 6, 4], 2) == 5\nassert find_kth_largest_element([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4\n"
    ))

    # 40. Top K Frequent Elements
    tasks.append(create_task(
        name="top_k_frequent_elements",
        signature="def top_k_frequent_elements(nums: list[int], k: int) -> list[int]:",
        doc_desc="Returns the k most frequent elements in nums, sorted by frequency descending.",
        doctests=[">>> top_k_frequent_elements([1, 1, 1, 2, 2, 3], 2)", "[1, 2]"],
        hint="Use collections.Counter.most_common(k) and extract element keys.",
        solution="    from collections import Counter\n    return [item[0] for item in Counter(nums).most_common(k)]",
        test="assert top_k_frequent_elements([1, 1, 1, 2, 2, 3], 2) == [1, 2]\nassert top_k_frequent_elements([1], 1) == [1]\n"
    ))

    # 41. Sort Colors (Dutch National Flag)
    tasks.append(create_task(
        name="sort_colors_dutch_flag",
        signature="def sort_colors_dutch_flag(nums: list[int]) -> list[int]:",
        doc_desc="Sorts array containing only 0s, 1s, and 2s in-place/in one pass returning the sorted list.",
        doctests=[">>> sort_colors_dutch_flag([2, 0, 2, 1, 1, 0])", "[0, 0, 1, 1, 2, 2]"],
        hint="Three pointers: low, mid = 0, 0; high = len - 1. Swap nums[low], nums[mid] if 0; swap mid, high if 2; else mid += 1.",
        solution="    arr = list(nums)\n    low, mid, high = 0, 0, len(arr) - 1\n    while mid <= high:\n        if arr[mid] == 0:\n            arr[low], arr[mid] = arr[mid], arr[low]\n            low += 1\n            mid += 1\n        elif arr[mid] == 1:\n            mid += 1\n        else:\n            arr[mid], arr[high] = arr[high], arr[mid]\n            high -= 1\n    return arr",
        test="assert sort_colors_dutch_flag([2, 0, 2, 1, 1, 0]) == [0, 0, 1, 1, 2, 2]\nassert sort_colors_dutch_flag([2, 0, 1]) == [0, 1, 2]\nassert sort_colors_dutch_flag([]) == []\n"
    ))

    # 42. Rotate Array K Steps
    tasks.append(create_task(
        name="rotate_array_k_steps",
        signature="def rotate_array_k_steps(nums: list[int], k: int) -> list[int]:",
        doc_desc="Rotates an array to the right by k non-negative steps.",
        doctests=[">>> rotate_array_k_steps([1, 2, 3, 4, 5, 6, 7], 3)", "[5, 6, 7, 1, 2, 3, 4]"],
        hint="Handle k %= len(nums). Return nums[-k:] + nums[:-k] if nums else [].",
        solution="    if not nums:\n        return []\n    k %= len(nums)\n    if k == 0:\n        return list(nums)\n    return nums[-k:] + nums[:-k]",
        test="assert rotate_array_k_steps([1, 2, 3, 4, 5, 6, 7], 3) == [5, 6, 7, 1, 2, 3, 4]\nassert rotate_array_k_steps([-1, -100, 3, 99], 2) == [3, 99, -1, -100]\nassert rotate_array_k_steps([1, 2], 3) == [2, 1]\nassert rotate_array_k_steps([], 5) == []\n"
    ))

    # 43. Max Subarray Sum Kadane
    tasks.append(create_task(
        name="max_subarray_sum_kadane",
        signature="def max_subarray_sum_kadane(nums: list[int]) -> int:",
        doc_desc="Finds the maximum sum of a contiguous non-empty subarray using Kadane's algorithm.",
        doctests=[">>> max_subarray_sum_kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4])", "6"],
        hint="Track curr_max = max(n, curr_max + n) and global_max = max(global_max, curr_max).",
        solution="    if not nums:\n        raise ValueError('Array must not be empty')\n    curr_max = global_max = nums[0]\n    for n in nums[1:]:\n        curr_max = max(n, curr_max + n)\n        global_max = max(global_max, curr_max)\n    return global_max",
        test="assert max_subarray_sum_kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\nassert max_subarray_sum_kadane([1]) == 1\nassert max_subarray_sum_kadane([5, 4, -1, 7, 8]) == 23\nassert max_subarray_sum_kadane([-3, -2, -1]) == -1\n"
    ))

    # 44. Product of Array Except Self
    tasks.append(create_task(
        name="product_except_self",
        signature="def product_except_self(nums: list[int]) -> list[int]:",
        doc_desc="Returns array where res[i] is the product of all elements of nums except nums[i] in O(n) time without division.",
        doctests=[">>> product_except_self([1, 2, 3, 4])", "[24, 12, 8, 6]"],
        hint="Compute prefix products in first pass left-to-right, then suffix products in second pass right-to-left.",
        solution="    n = len(nums)\n    res = [1] * n\n    prefix = 1\n    for i in range(n):\n        res[i] = prefix\n        prefix *= nums[i]\n    suffix = 1\n    for i in range(n - 1, -1, -1):\n        res[i] *= suffix\n        suffix *= nums[i]\n    return res",
        test="assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]\nassert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]\n"
    ))

    # 45. Spiral Order Matrix
    tasks.append(create_task(
        name="spiral_order_matrix",
        signature="def spiral_order_matrix(matrix: list[list[int]]) -> list[int]:",
        doc_desc="Returns all elements of an m x n 2D matrix in spiral order.",
        doctests=[">>> spiral_order_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])", "[1, 2, 3, 6, 9, 8, 7, 4, 5]"],
        hint="Track four boundaries: top, bottom, left, right. Traverse top row, right col, bottom row, left col, shrinking bounds.",
        solution="    if not matrix or not matrix[0]:\n        return []\n    res = []\n    top, bottom = 0, len(matrix) - 1\n    left, right = 0, len(matrix[0]) - 1\n    while top <= bottom and left <= right:\n        for j in range(left, right + 1):\n            res.append(matrix[top][j])\n        top += 1\n        for i in range(top, bottom + 1):\n            res.append(matrix[i][right])\n        right -= 1\n        if top <= bottom:\n            for j in range(right, left - 1, -1):\n                res.append(matrix[bottom][j])\n            bottom -= 1\n        if left <= right:\n            for i in range(bottom, top - 1, -1):\n                res.append(matrix[i][left])\n            left += 1\n    return res",
        test="assert spiral_order_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]\nassert spiral_order_matrix([[1]]) == [1]\nassert spiral_order_matrix([]) == []\n"
    ))

    # 46. Matrix Transpose
    tasks.append(create_task(
        name="matrix_transpose",
        signature="def matrix_transpose(matrix: list[list[Any]]) -> list[list[Any]]:",
        doc_desc="Transposes an m x n 2D matrix into an n x m matrix.",
        doctests=[">>> matrix_transpose([[1, 2], [3, 4], [5, 6]])", "[[1, 3, 5], [2, 4, 6]]"],
        hint="Use list comprehension with zip(*matrix) or [[matrix[i][j] for i in range(rows)] for j in range(cols)].",
        solution="    if not matrix:\n        return []\n    return [list(row) for row in zip(*matrix)]",
        test="assert matrix_transpose([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]\nassert matrix_transpose([[1]]) == [[1]]\nassert matrix_transpose([]) == []\n"
    ))

    # 47. Rotate Matrix 90 Clockwise
    tasks.append(create_task(
        name="rotate_matrix_90_clockwise",
        signature="def rotate_matrix_90_clockwise(matrix: list[list[int]]) -> list[list[int]]:",
        doc_desc="Rotates an n x n 2D matrix 90 degrees clockwise.",
        doctests=[">>> rotate_matrix_90_clockwise([[1, 2], [3, 4]])", "[[3, 1], [4, 2]]"],
        hint="Transpose the matrix, then reverse each row.",
        solution="    if not matrix:\n        return []\n    return [list(reversed(col)) for col in zip(*matrix)]",
        test="assert rotate_matrix_90_clockwise([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]\nassert rotate_matrix_90_clockwise([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]\n"
    ))

    # 48. Set Matrix Zeroes
    tasks.append(create_task(
        name="set_matrix_zeroes",
        signature="def set_matrix_zeroes(matrix: list[list[int]]) -> list[list[int]]:",
        doc_desc="If an element in an m x n matrix is 0, sets its entire row and column to 0, returning the modified matrix.",
        doctests=[">>> set_matrix_zeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]])", "[[1, 0, 1], [0, 0, 0], [1, 0, 1]]"],
        hint="Record rows and cols that contain 0 in sets, then update cells where i in zero_rows or j in zero_cols.",
        solution="    if not matrix:\n        return []\n    m, n = len(matrix), len(matrix[0])\n    res = [list(row) for row in matrix]\n    zero_rows = {i for i in range(m) for j in range(n) if matrix[i][j] == 0}\n    zero_cols = {j for i in range(m) for j in range(n) if matrix[i][j] == 0}\n    for i in range(m):\n        for j in range(n):\n            if i in zero_rows or j in zero_cols:\n                res[i][j] = 0\n    return res",
        test="assert set_matrix_zeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]\nassert set_matrix_zeroes([[0, 1]]) == [[0, 0]]\n"
    ))

    # 49. Search 2D Sorted Matrix
    tasks.append(create_task(
        name="search_2d_sorted_matrix",
        signature="def search_2d_sorted_matrix(matrix: list[list[int]], target: int) -> bool:",
        doc_desc="Determines if target exists in m x n matrix where each row is sorted and first int of row > last int of previous.",
        doctests=[">>> search_2d_sorted_matrix([[1, 3, 5, 7], [10, 11, 16, 20]], 3)", "True"],
        hint="Treat as 1D array of size m * n: mid maps to row mid // n, col mid % n.",
        solution="    if not matrix or not matrix[0]:\n        return False\n    m, n = len(matrix), len(matrix[0])\n    low, high = 0, m * n - 1\n    while low <= high:\n        mid = (low + high) // 2\n        val = matrix[mid // n][mid % n]\n        if val == target:\n            return True\n        elif val < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return False",
        test="assert search_2d_sorted_matrix([[1, 3, 5, 7], [10, 11, 16, 20]], 3) is True\nassert search_2d_sorted_matrix([[1, 3, 5, 7], [10, 11, 16, 20]], 13) is False\nassert search_2d_sorted_matrix([], 1) is False\n"
    ))

    # 50. Number of Islands Grid
    tasks.append(create_task(
        name="number_of_islands_grid",
        signature="def number_of_islands_grid(grid: list[list[str]]) -> int:",
        doc_desc="Counts the number of connected 4-directional islands of '1's in a 2D binary grid.",
        doctests=[">>> number_of_islands_grid([['1', '1', '0'], ['1', '1', '0'], ['0', '0', '1']])", "2"],
        hint="Iterate cells: when '1' found, increment count and run BFS/DFS turning all connected '1's into '0'.",
        solution="    if not grid:\n        return 0\n    m, n = len(grid), len(grid[0])\n    g = [list(row) for row in grid]\n    count = 0\n    for i in range(m):\n        for j in range(n):\n            if g[i][j] == '1':\n                count += 1\n                stack = [(i, j)]\n                g[i][j] = '0'\n                while stack:\n                    r, c = stack.pop()\n                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:\n                        nr, nc = r + dr, c + dc\n                        if 0 <= nr < m and 0 <= nc < n and g[nr][nc] == '1':\n                            g[nr][nc] = '0'\n                            stack.append((nr, nc))\n    return count",
        test="assert number_of_islands_grid([['1', '1', '0'], ['1', '1', '0'], ['0', '0', '1']]) == 2\nassert number_of_islands_grid([['1', '1', '1']]) == 1\nassert number_of_islands_grid([['0']]) == 0\n"
    ))

    # 51. Shortest Path in Binary Matrix
    tasks.append(create_task(
        name="shortest_path_binary_matrix",
        signature="def shortest_path_binary_matrix(grid: list[list[int]]) -> int:",
        doc_desc="Returns length of shortest 8-directional clear path from top-left (0,0) to bottom-right (n-1,n-1) in grid with 0s as clear.",
        doctests=[">>> shortest_path_binary_matrix([[0, 1], [1, 0]])", "2"],
        hint="Use BFS starting at (0, 0) with distance 1. Explore all 8 adjacent cells where cell == 0.",
        solution="    n = len(grid)\n    if grid[0][0] != 0 or grid[n-1][n-1] != 0:\n        return -1\n    from collections import deque\n    q = deque([(0, 0, 1)])\n    visited = {(0, 0)}\n    while q:\n        r, c, d = q.popleft()\n        if r == n - 1 and c == n - 1:\n            return d\n        for dr in (-1, 0, 1):\n            for dc in (-1, 0, 1):\n                if dr == 0 and dc == 0: continue\n                nr, nc = r + dr, c + dc\n                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and (nr, nc) not in visited:\n                    visited.add((nr, nc))\n                    q.append((nr, nc, d + 1))\n    return -1",
        test="assert shortest_path_binary_matrix([[0, 1], [1, 0]]) == 2\nassert shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4\nassert shortest_path_binary_matrix([[1, 0], [0, 0]]) == -1\n"
    ))

    # 52. Word Search Grid
    tasks.append(create_task(
        name="word_search_grid",
        signature="def word_search_grid(board: list[list[str]], word: str) -> bool:",
        doc_desc="Checks if word exists in 2D character grid constructed from 4-directionally adjacent cells.",
        doctests=[">>> word_search_grid([['A', 'B', 'C'], ['S', 'F', 'C'], ['A', 'D', 'E']], 'ABCCED')", "True"],
        hint="Backtracking DFS: mark cell visited with '#', recurse in 4 directions for next char index, restore cell on return.",
        solution="    m, n = len(board), len(board[0])\n    b = [list(row) for row in board]\n    def dfs(r, c, k):\n        if k == len(word): return True\n        if not (0 <= r < m and 0 <= c < n and b[r][c] == word[k]): return False\n        temp = b[r][c]\n        b[r][c] = '#'\n        res = (dfs(r+1, c, k+1) or dfs(r-1, c, k+1) or dfs(r, c+1, k+1) or dfs(r, c-1, k+1))\n        b[r][c] = temp\n        return res\n    for i in range(m):\n        for j in range(n):\n            if dfs(i, j, 0):\n                return True\n    return False",
        test="b = [['A', 'B', 'C'], ['S', 'E', 'E'], ['A', 'D', 'E']]\nassert word_search_grid(b, 'ABCCED') is False\nassert word_search_grid(b, 'SEE') is True\nassert word_search_grid(b, 'ABCB') is False\n"
    ))

    # 53. Interval Intersection
    tasks.append(create_task(
        name="interval_intersection",
        signature="def interval_intersection(first: list[list[int]], second: list[list[int]]) -> list[list[int]]:",
        doc_desc="Finds intersection of two sorted, disjoint interval lists.",
        doctests=[">>> interval_intersection([[0, 2], [5, 10]], [[1, 5], [8, 12]])", "[[1, 2], [5, 5], [8, 10]]"],
        hint="Two pointers i and j. Start = max(first[i][0], second[j][0]), end = min(first[i][1], second[j][1]). Advance interval with smaller end.",
        solution="    i, j = 0, 0\n    res = []\n    while i < len(first) and j < len(second):\n        lo = max(first[i][0], second[j][0])\n        hi = min(first[i][1], second[j][1])\n        if lo <= hi:\n            res.append([lo, hi])\n        if first[i][1] < second[j][1]:\n            i += 1\n        else:\n            j += 1\n    return res",
        test="assert interval_intersection([[0, 2], [5, 10]], [[1, 5], [8, 12]]) == [[1, 2], [5, 5], [8, 10]]\nassert interval_intersection([[1, 3]], [[5, 9]]) == []\nassert interval_intersection([], []) == []\n"
    ))

    # 54. Insert Interval
    tasks.append(create_task(
        name="insert_interval",
        signature="def insert_interval(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:",
        doc_desc="Inserts new_interval into sorted non-overlapping intervals, merging if necessary.",
        doctests=[">>> insert_interval([[1, 3], [6, 9]], [2, 5])", "[[1, 5], [6, 9]]"],
        hint="Append intervals before new_interval. Merge overlapping intervals into new_interval by updating start and end. Append remaining.",
        solution="    res = []\n    i = 0\n    n = len(intervals)\n    while i < n and intervals[i][1] < new_interval[0]:\n        res.append(intervals[i])\n        i += 1\n    while i < n and intervals[i][0] <= new_interval[1]:\n        new_interval = [min(new_interval[0], intervals[i][0]), max(new_interval[1], intervals[i][1])]\n        i += 1\n    res.append(new_interval)\n    while i < n:\n        res.append(intervals[i])\n        i += 1\n    return res",
        test="assert insert_interval([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]\nassert insert_interval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [[1, 2], [3, 10], [12, 16]]\nassert insert_interval([], [5, 7]) == [[5, 7]]\n"
    ))

    # 55. Longest Increasing Subsequence Length
    tasks.append(create_task(
        name="longest_increasing_subsequence_length",
        signature="def longest_increasing_subsequence_length(nums: list[int]) -> int:",
        doc_desc="Calculates the length of the longest strictly increasing subsequence in O(n log n) time.",
        doctests=[">>> longest_increasing_subsequence_length([10, 9, 2, 5, 3, 7, 101, 18])", "4"],
        hint="Maintain array tails. For each x in nums, use bisect.bisect_left(tails, x) to replace or append to tails.",
        solution="    import bisect\n    tails = []\n    for x in nums:\n        idx = bisect.bisect_left(tails, x)\n        if idx == len(tails):\n            tails.append(x)\n        else:\n            tails[idx] = x\n    return len(tails)",
        test="assert longest_increasing_subsequence_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4\nassert longest_increasing_subsequence_length([0, 1, 0, 3, 2, 3]) == 4\nassert longest_increasing_subsequence_length([7, 7, 7]) == 1\nassert longest_increasing_subsequence_length([]) == 0\n"
    ))

    # 56. Coin Change Min Coins
    tasks.append(create_task(
        name="coin_change_min_coins",
        signature="def coin_change_min_coins(coins: list[int], amount: int) -> int:",
        doc_desc="Finds fewest number of coins needed to make up amount, or -1 if amount cannot be formed.",
        doctests=[">>> coin_change_min_coins([1, 2, 5], 11)", "3"],
        hint="DP array of size amount + 1 initialized to inf. dp[0] = 0. For each coin, update dp[i] = min(dp[i], dp[i - coin] + 1).",
        solution="    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for c in coins:\n        for i in range(c, amount + 1):\n            dp[i] = min(dp[i], dp[i - c] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1",
        test="assert coin_change_min_coins([1, 2, 5], 11) == 3\nassert coin_change_min_coins([2], 3) == -1\nassert coin_change_min_coins([1], 0) == 0\n"
    ))

    # 57. Daily Temperatures Wait Days
    tasks.append(create_task(
        name="daily_temperatures_wait_days",
        signature="def daily_temperatures_wait_days(temperatures: list[int]) -> list[int]:",
        doc_desc="Returns array where ans[i] is days until a warmer temperature, or 0 if none.",
        doctests=[">>> daily_temperatures_wait_days([73, 74, 75, 71, 69, 72, 76, 73])", "[1, 1, 4, 2, 1, 1, 0, 0]"],
        hint="Use a monotonic stack storing indices of temperatures. While current temp > stack top temp: pop prev index and set diff.",
        solution="    n = len(temperatures)\n    ans = [0] * n\n    stack = []\n    for i, t in enumerate(temperatures):\n        while stack and temperatures[stack[-1]] < t:\n            prev = stack.pop()\n            ans[prev] = i - prev\n        stack.append(i)\n    return ans",
        test="assert daily_temperatures_wait_days([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]\nassert daily_temperatures_wait_days([30, 40, 50]) == [1, 1, 0]\nassert daily_temperatures_wait_days([30, 30]) == [0, 0]\n"
    ))

    # 58. Max Consecutive Ones Flip One
    tasks.append(create_task(
        name="max_consecutive_ones_flip_one",
        signature="def max_consecutive_ones_flip_one(nums: list[int]) -> int:",
        doc_desc="Finds maximum consecutive 1s in binary array if at most one 0 can be flipped to 1.",
        doctests=[">>> max_consecutive_ones_flip_one([1, 0, 1, 1, 0])", "4"],
        hint="Sliding window with two pointers left and right. Count zeros; while zeros > 1: if nums[left] == 0: zeros -= 1; left += 1.",
        solution="    left = 0\n    zeros = 0\n    max_len = 0\n    for right, val in enumerate(nums):\n        if val == 0:\n            zeros += 1\n        while zeros > 1:\n            if nums[left] == 0:\n                zeros -= 1\n            left += 1\n        max_len = max(max_len, right - left + 1)\n    return max_len",
        test="assert max_consecutive_ones_flip_one([1, 0, 1, 1, 0]) == 4\nassert max_consecutive_ones_flip_one([1, 0, 1, 1, 0, 1]) == 4\nassert max_consecutive_ones_flip_one([0, 0]) == 1\n"
    ))

    # 59. Find Peak Element 1D
    tasks.append(create_task(
        name="find_peak_element_1d",
        signature="def find_peak_element_1d(nums: list[int]) -> int:",
        doc_desc="Returns the index of any peak element (strictly greater than neighbors) in O(log n) time.",
        doctests=[">>> find_peak_element_1d([1, 2, 3, 1])", "2"],
        hint="Binary search: if nums[mid] < nums[mid + 1] search right (low = mid + 1); else search left (high = mid).",
        solution="    low, high = 0, len(nums) - 1\n    while low < high:\n        mid = (low + high) // 2\n        if nums[mid] < nums[mid + 1]:\n            low = mid + 1\n        else:\n            high = mid\n    return low",
        test="assert find_peak_element_1d([1, 2, 3, 1]) == 2\nassert find_peak_element_1d([1]) == 0\nassert find_peak_element_1d([1, 2]) == 1\n"
    ))

    # 60. Min Depth of Binary Tree
    tasks.append(create_task(
        name="binary_tree_min_depth",
        signature="def binary_tree_min_depth(root: dict | None) -> int:",
        doc_desc="Calculates minimum depth (shortest path to a leaf node) in binary tree.",
        doctests=[">>> binary_tree_min_depth({'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': None})", "2"],
        hint="If root is None return 0. If left is None return 1 + depth(right); if right is None return 1 + depth(left); else 1 + min(left, right).",
        solution="    if not root:\n        return 0\n    left = binary_tree_min_depth(root.get('left'))\n    right = binary_tree_min_depth(root.get('right'))\n    if not root.get('left') or not root.get('right'):\n        return 1 + left + right\n    return 1 + min(left, right)",
        test="t = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': None}\nassert binary_tree_min_depth(t) == 2\nassert binary_tree_min_depth(None) == 0\n"
    ))

    # 61. Sum Root to Leaf Numbers
    tasks.append(create_task(
        name="sum_root_to_leaf_numbers",
        signature="def sum_root_to_leaf_numbers(root: dict | None) -> int:",
        doc_desc="Returns sum of all numbers formed by root-to-leaf paths where each node contains a digit 0-9.",
        doctests=[">>> sum_root_to_leaf_numbers({'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}})", "25"],
        hint="DFS passing current value curr * 10 + node['val']. At leaf return curr; else sum left and right recursion.",
        solution="    def dfs(node, curr):\n        if not node:\n            return 0\n        curr = curr * 10 + node['val']\n        if not node.get('left') and not node.get('right'):\n            return curr\n        return dfs(node.get('left'), curr) + dfs(node.get('right'), curr)\n    return dfs(root, 0)",
        test="t = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}\nassert sum_root_to_leaf_numbers(t) == 25\nassert sum_root_to_leaf_numbers(None) == 0\n"
    ))

    # 62. Invert Dictionary Keys Values
    tasks.append(create_task(
        name="invert_dict_multivalue",
        signature="def invert_dict_multivalue(d: dict[str, Any]) -> dict[Any, list[str]]:",
        doc_desc="Inverts a dictionary mapping each value to a list of keys having that value, with keys sorted.",
        doctests=[">>> invert_dict_multivalue({'a': 1, 'b': 2, 'c': 1})", "{1: ['a', 'c'], 2: ['b']}"],
        hint="Use collections.defaultdict(list). Iterate k, v in d.items(), append k to res[v], then sort each list.",
        solution="    from collections import defaultdict\n    res = defaultdict(list)\n    for k, v in d.items():\n        res[v].append(k)\n    return {k: sorted(v) for k, v in res.items()}",
        test="assert invert_dict_multivalue({'a': 1, 'b': 2, 'c': 1}) == {1: ['a', 'c'], 2: ['b']}\nassert invert_dict_multivalue({}) == {}\n"
    ))

    # 63. Deep Flatten Nested List
    tasks.append(create_task(
        name="deep_flatten_nested_list",
        signature="def deep_flatten_nested_list(lst: list) -> list:",
        doc_desc="Flattens an arbitrarily nested list into a single flat list.",
        doctests=[">>> deep_flatten_nested_list([1, [2, [3, 4], 5], 6])", "[1, 2, 3, 4, 5, 6]"],
        hint="Recursively traverse: if element is list, extend with deep_flatten(element); else append element.",
        solution="    res = []\n    for item in lst:\n        if isinstance(item, list):\n            res.extend(deep_flatten_nested_list(item))\n        else:\n            res.append(item)\n    return res",
        test="assert deep_flatten_nested_list([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]\nassert deep_flatten_nested_list([]) == []\nassert deep_flatten_nested_list([[], [[]]]) == []\n"
    ))

    # 64. Circular Queue Is Empty Full Check
    tasks.append(create_task(
        name="BoundedStack",
        signature="class BoundedStack:\n    def __init__(self, max_capacity: int):\n        self.cap = max_capacity\n        self.items = []",
        doc_desc="A capacity-bounded LIFO stack rejecting push operations when full.",
        doctests=[">>> bs = BoundedStack(2); bs.push(1); bs.push(2); bs.push(3)", "False"],
        hint="If len(items) < cap: append item and return True. Pop returns item or None. is_full returns len == cap.",
        solution="    def push(self, item: Any) -> bool:\n        if len(self.items) < self.cap:\n            self.items.append(item)\n            return True\n        return False\n    def pop(self) -> Any | None:\n        return self.items.pop() if self.items else None\n    def is_full(self) -> bool: return len(self.items) >= self.cap\n    def is_empty(self) -> bool: return len(self.items) == 0",
        test="bs = BoundedStack(2)\nassert bs.push(1) is True\nassert bs.push(2) is True\nassert bs.push(3) is False\nassert bs.is_full() is True\nassert bs.pop() == 2\nassert bs.is_full() is False\n"
    ))

    return tasks
