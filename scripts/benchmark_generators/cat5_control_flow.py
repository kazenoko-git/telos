"""
Category 5: Control Flow & Loops (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_control_flow_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Traffic Light State Machine
    tasks.append(create_task(
        name="simulate_traffic_light_sequence",
        signature="def simulate_traffic_light_sequence(events: list[str]) -> list[str]:",
        doc_desc="Simulates traffic light state machine starting at 'RED'. 'TICK' transitions RED->GREEN->YELLOW->RED.",
        doctests=[">>> simulate_traffic_light_sequence(['TICK', 'TICK'])", "['GREEN', 'YELLOW']"],
        hint="Loop over events: if event == 'TICK', cycle state RED -> GREEN -> YELLOW -> RED and record current state.",
        solution="    state = 'RED'\n    res = []\n    transitions = {'RED': 'GREEN', 'GREEN': 'YELLOW', 'YELLOW': 'RED'}\n    for e in events:\n        if e == 'TICK':\n            state = transitions[state]\n            res.append(state)\n    return res",
        test="assert simulate_traffic_light_sequence(['TICK', 'TICK']) == ['GREEN', 'YELLOW']\nassert simulate_traffic_light_sequence(['TICK', 'TICK', 'TICK']) == ['GREEN', 'YELLOW', 'RED']\nassert simulate_traffic_light_sequence(['NOOP']) == []\n"
    ))

    # 2. Elevator Movement
    tasks.append(create_task(
        name="simulate_elevator_floors",
        signature="def simulate_elevator_floors(requests: list[int], start_floor: int = 1) -> list[int]:",
        doc_desc="Simulates an elevator moving floor by floor to service sequential target floor requests, recording visited floors.",
        doctests=[">>> simulate_elevator_floors([3, 1], 1)", "[2, 3, 2, 1]"],
        hint="From current floor to target floor: step = 1 if target > current else -1; step floor by floor and append each floor.",
        solution="    curr = start_floor\n    visited = []\n    for target in requests:\n        step = 1 if target > curr else -1\n        while curr != target:\n            curr += step\n            visited.append(curr)\n    return visited",
        test="assert simulate_elevator_floors([3, 1], 1) == [2, 3, 2, 1]\nassert simulate_elevator_floors([1], 1) == []\nassert simulate_elevator_floors([4], 2) == [3, 4]\n"
    ))

    # 3. Conway's Game of Life 1D
    tasks.append(create_task(
        name="conway_game_of_life_1d",
        signature="def conway_game_of_life_1d(cells: list[int], steps: int = 1) -> list[int]:",
        doc_desc="Simulates 1D cellular automaton for steps steps: cell survives/born if exactly one neighbor is 1 (boundary is 0).",
        doctests=[">>> conway_game_of_life_1d([1, 0, 1], 1)", "[0, 0, 0]"],
        hint="In each step, next_cells[i] = 1 if (left_neighbor + right_neighbor == 1) else 0. Boundary neighbors are 0.",
        solution="    curr = list(cells)\n    n = len(curr)\n    for _ in range(steps):\n        nxt = [0] * n\n        for i in range(n):\n            left = curr[i - 1] if i > 0 else 0\n            right = curr[i + 1] if i < n - 1 else 0\n            nxt[i] = 1 if (left + right == 1) else 0\n        curr = nxt\n    return curr",
        test="assert conway_game_of_life_1d([1, 0, 1], 1) == [0, 0, 0]\nassert conway_game_of_life_1d([0, 1, 0], 1) == [1, 0, 1]\nassert conway_game_of_life_1d([0, 0], 2) == [0, 0]\n"
    ))

    # 4. Collatz Trajectory Recorder
    tasks.append(create_task(
        name="collatz_trajectory_recorder",
        signature="def collatz_trajectory_recorder(start: int) -> list[int]:",
        doc_desc="Records the full sequence of numbers from start down to 1 in the 3n + 1 problem.",
        doctests=[">>> collatz_trajectory_recorder(6)", "[6, 3, 10, 5, 16, 8, 4, 2, 1]"],
        hint="Start trajectory with [start]. While start > 1: start = start // 2 if start % 2 == 0 else 3 * start + 1; append start.",
        solution="    if start <= 0:\n        return []\n    res = [start]\n    while start > 1:\n        start = start // 2 if start % 2 == 0 else 3 * start + 1\n        res.append(start)\n    return res",
        test="assert collatz_trajectory_recorder(6) == [6, 3, 10, 5, 16, 8, 4, 2, 1]\nassert collatz_trajectory_recorder(1) == [1]\nassert collatz_trajectory_recorder(2) == [2, 1]\n"
    ))

    # 5. Billiards Bounce 1D
    tasks.append(create_task(
        name="simulate_billiards_bounce_1d",
        signature="def simulate_billiards_bounce_1d(length: int, steps: int, start_pos: int = 0, start_dir: int = 1) -> int:",
        doc_desc="Simulates ball moving on 1D track 0..length with direction +1 or -1, reflecting at walls 0 and length.",
        doctests=[">>> simulate_billiards_bounce_1d(5, 7, 0, 1)", "3"],
        hint="Loop steps: pos += direction. If pos == length: direction = -1; elif pos == 0: direction = 1.",
        solution="    pos = start_pos\n    direction = start_dir\n    for _ in range(steps):\n        pos += direction\n        if pos == length:\n            direction = -1\n        elif pos == 0:\n            direction = 1\n    return pos",
        test="assert simulate_billiards_bounce_1d(5, 7, 0, 1) == 3\nassert simulate_billiards_bounce_1d(5, 5, 0, 1) == 5\nassert simulate_billiards_bounce_1d(5, 10, 0, 1) == 0\n"
    ))

    # 6. Token Bucket Simulation
    tasks.append(create_task(
        name="simulate_token_bucket_requests",
        signature="def simulate_token_bucket_requests(capacity: int, refill_rate: int, requests: list[int]) -> list[bool]:",
        doc_desc="Simulates token bucket for each step in requests. At step start, refills tokens up to capacity, then evaluates request.",
        doctests=[">>> simulate_token_bucket_requests(3, 1, [3, 2, 1])", "[True, False, True]"],
        hint="tokens = capacity. For req in requests: tokens = min(capacity, tokens + refill_rate); if tokens >= req: tokens -= req, True; else False.",
        solution="    tokens = capacity\n    res = []\n    for req in requests:\n        tokens = min(capacity, tokens + refill_rate)\n        if tokens >= req:\n            tokens -= req\n            res.append(True)\n        else:\n            res.append(False)\n    return res",
        test="assert simulate_token_bucket_requests(3, 1, [3, 2, 1]) == [True, False, True]\nassert simulate_token_bucket_requests(2, 0, [1, 1, 1]) == [True, True, False]\nassert simulate_token_bucket_requests(5, 1, []) == []\n"
    ))

    # 7. Turtle Grid Path
    tasks.append(create_task(
        name="simulate_turtle_grid_path",
        signature="def simulate_turtle_grid_path(commands: list[str]) -> tuple[int, int]:",
        doc_desc="Simulates turtle on 2D grid starting at (0, 0) facing 'N'. Commands: 'F' (forward 1), 'L' (turn left), 'R' (turn right).",
        doctests=[">>> simulate_turtle_grid_path(['F', 'R', 'F', 'F'])", "(2, 1)"],
        hint="Track (x, y) and dir_idx in ['N', 'E', 'S', 'W']. 'F' moves by direction delta. 'L' sets (dir_idx - 1) % 4, 'R' (dir_idx + 1) % 4.",
        solution="    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W\n    d_idx = 0\n    x, y = 0, 0\n    for cmd in commands:\n        if cmd == 'F':\n            dx, dy = dirs[d_idx]\n            x += dx\n            y += dy\n        elif cmd == 'L':\n            d_idx = (d_idx - 1) % 4\n        elif cmd == 'R':\n            d_idx = (d_idx + 1) % 4\n    return (x, y)",
        test="assert simulate_turtle_grid_path(['F', 'R', 'F', 'F']) == (2, 1)\nassert simulate_turtle_grid_path(['F', 'L', 'F', 'L', 'F', 'L', 'F']) == (0, 0)\nassert simulate_turtle_grid_path([]) == (0, 0)\n"
    ))

    # 8. Vending Coin Exchange
    tasks.append(create_task(
        name="calculate_vending_change_coins",
        signature="def calculate_vending_change_coins(cents: int, denominations: list[int] = [25, 10, 5, 1]) -> dict[int, int]:",
        doc_desc="Calculates greedy coin change distribution returning {denomination: count} for all denominations used.",
        doctests=[">>> calculate_vending_change_coins(67)", "{25: 2, 10: 1, 5: 1, 1: 2}"],
        hint="Sort denominations descending. For each coin: count = cents // coin; if count > 0: res[coin] = count; cents %= coin.",
        solution="    res = {}\n    for coin in sorted(denominations, reverse=True):\n        if cents >= coin:\n            count = cents // coin\n            res[coin] = count\n            cents %= coin\n    return res",
        test="assert calculate_vending_change_coins(67) == {25: 2, 10: 1, 5: 1, 1: 2}\nassert calculate_vending_change_coins(25) == {25: 1}\nassert calculate_vending_change_coins(0) == {}\n"
    ))

    # 9. Blackjack Hand Value
    tasks.append(create_task(
        name="calculate_blackjack_hand_score",
        signature="def calculate_blackjack_hand_score(cards: list[str]) -> int:",
        doc_desc="Calculates Blackjack hand score where 'J', 'Q', 'K' are 10, '2'..'10' are face value, and 'A' is 11 unless busting (>21) then 1.",
        doctests=[">>> calculate_blackjack_hand_score(['A', 'K'])", "21", ">>> calculate_blackjack_hand_score(['A', 'A', '9'])", "21"],
        hint="Count aces. Add 10 for face cards, 11 for aces, int(card) for numbers. While total > 21 and aces > 0: total -= 10, aces -= 1.",
        solution="    total = 0\n    aces = 0\n    for c in cards:\n        if c in ('J', 'Q', 'K'): total += 10\n        elif c == 'A':\n            total += 11\n            aces += 1\n        else:\n            total += int(c)\n    while total > 21 and aces > 0:\n        total -= 10\n        aces -= 1\n    return total",
        test="assert calculate_blackjack_hand_score(['A', 'K']) == 21\nassert calculate_blackjack_hand_score(['A', 'A', '9']) == 21\nassert calculate_blackjack_hand_score(['10', '7', '8']) == 25\nassert calculate_blackjack_hand_score(['A', '5', '5']) == 21\n"
    ))

    # 10. Round Robin Tournament
    tasks.append(create_task(
        name="simulate_round_robin_tournament",
        signature="def simulate_round_robin_tournament(matches: list[tuple[str, str, int, int]]) -> dict[str, int]:",
        doc_desc="Simulates tournament from (teamA, teamB, scoreA, scoreB) where win awards 3 points, draw 1 point, loss 0.",
        doctests=[">>> simulate_round_robin_tournament([('A', 'B', 2, 1), ('B', 'C', 1, 1)])", "{'A': 3, 'B': 1, 'C': 1}"],
        hint="Initialize points dict for both teams. If scoreA > scoreB: A+=3; elif scoreA < scoreB: B+=3; else: A+=1, B+=1.",
        solution="    points = {}\n    for a, b, sa, sb in matches:\n        points.setdefault(a, 0)\n        points.setdefault(b, 0)\n        if sa > sb:\n            points[a] += 3\n        elif sb > sa:\n            points[b] += 3\n        else:\n            points[a] += 1\n            points[b] += 1\n    return points",
        test="assert simulate_round_robin_tournament([('A', 'B', 2, 1), ('B', 'C', 1, 1)]) == {'A': 3, 'B': 1, 'C': 1}\nassert simulate_round_robin_tournament([]) == {}\n"
    ))

    # 11. Container With Most Water
    tasks.append(create_task(
        name="container_with_most_water",
        signature="def container_with_most_water(heights: list[int]) -> int:",
        doc_desc="Finds two lines that together with x-axis form a container holding the most water.",
        doctests=[">>> container_with_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7])", "49"],
        hint="Two pointers left=0, right=len-1. Area = (right - left) * min(h[left], h[right]). Advance pointer with smaller height.",
        solution="    left, right = 0, len(heights) - 1\n    max_water = 0\n    while left < right:\n        w = right - left\n        h = min(heights[left], heights[right])\n        max_water = max(max_water, w * h)\n        if heights[left] < heights[right]:\n            left += 1\n        else:\n            right -= 1\n    return max_water",
        test="assert container_with_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49\nassert container_with_most_water([1, 1]) == 1\nassert container_with_most_water([4, 3, 2, 1, 4]) == 16\n"
    ))

    # 12. Monotonically Increasing Subarrays Count
    tasks.append(create_task(
        name="count_increasing_subarrays",
        signature="def count_increasing_subarrays(nums: list[int]) -> int:",
        doc_desc="Counts the number of strictly increasing contiguous subarrays of length at least 2.",
        doctests=[">>> count_increasing_subarrays([1, 2, 3])", "3"],
        hint="Track streak length where nums[i] > nums[i-1]. If streak is L, add L * (L - 1) // 2 for each maximal streak.",
        solution="    if len(nums) < 2:\n        return 0\n    total = 0\n    streak = 1\n    for i in range(1, len(nums)):\n        if nums[i] > nums[i - 1]:\n            streak += 1\n        else:\n            total += streak * (streak - 1) // 2\n            streak = 1\n    total += streak * (streak - 1) // 2\n    return total",
        test="assert count_increasing_subarrays([1, 2, 3]) == 3\nassert count_increasing_subarrays([1, 4, 2, 5, 3]) == 2\nassert count_increasing_subarrays([5, 4, 3, 2, 1]) == 0\n"
    ))

    # 13. Bank Transactions With Overdraft
    tasks.append(create_task(
        name="process_transactions_with_overdraft",
        signature="def process_transactions_with_overdraft(initial_balance: float, transactions: list[float], max_overdraft: float = 100.0, fee: float = 20.0) -> tuple[float, list[bool]]:",
        doc_desc="Processes transaction amounts (positive credit, negative debit). If debit causes balance < 0 within max_overdraft, apply fee.",
        doctests=[">>> process_transactions_with_overdraft(50.0, [-70.0, -100.0])", "(-40.0, [True, False])"],
        hint="For debit: if balance + amount >= -max_overdraft: if balance + amount < 0 (went into overdraft): balance -= fee; balance += amount, success; else reject.",
        solution="    balance = float(initial_balance)\n    status = []\n    for tx in transactions:\n        if tx >= 0:\n            balance += tx\n            status.append(True)\n        else:\n            new_bal = balance + tx\n            fee_to_apply = fee if new_bal < 0 else 0.0\n            if new_bal - fee_to_apply >= -max_overdraft:\n                balance = new_bal - fee_to_apply\n                status.append(True)\n            else:\n                status.append(False)\n    return (balance, status)",
        test="bal, st = process_transactions_with_overdraft(50.0, [-70.0, -100.0])\nassert st == [True, False]\nassert bal == -40.0\n"
    ))

    # 14. Spiral Grid Generator
    tasks.append(create_task(
        name="generate_spiral_matrix",
        signature="def generate_spiral_matrix(n: int) -> list[list[int]]:",
        doc_desc="Generates an n x n 2D matrix filled with elements 1 to n^2 in clockwise spiral order.",
        doctests=[">>> generate_spiral_matrix(3)", "[[1, 2, 3], [8, 9, 4], [7, 6, 5]]"],
        hint="Initialize n x n matrix with 0. Track top, bottom, left, right bounds and fill numbers 1..n^2.",
        solution="    matrix = [[0] * n for _ in range(n)]\n    num = 1\n    top, bottom = 0, n - 1\n    left, right = 0, n - 1\n    while top <= bottom and left <= right:\n        for j in range(left, right + 1):\n            matrix[top][j] = num\n            num += 1\n        top += 1\n        for i in range(top, bottom + 1):\n            matrix[i][right] = num\n            num += 1\n        right -= 1\n        if top <= bottom:\n            for j in range(right, left - 1, -1):\n                matrix[bottom][j] = num\n                num += 1\n            bottom -= 1\n        if left <= right:\n            for i in range(bottom, top - 1, -1):\n                matrix[i][left] = num\n                num += 1\n            left += 1\n    return matrix",
        test="assert generate_spiral_matrix(3) == [[1, 2, 3], [8, 9, 4], [7, 6, 5]]\nassert generate_spiral_matrix(1) == [[1]]\nassert generate_spiral_matrix(2) == [[1, 2], [4, 3]]\n"
    ))

    # 15. Snake Pattern Matrix Traversal
    tasks.append(create_task(
        name="snake_matrix_traversal",
        signature="def snake_matrix_traversal(matrix: list[list[int]]) -> list[int]:",
        doc_desc="Traverses matrix in snake order: even rows left-to-right, odd rows right-to-left.",
        doctests=[">>> snake_matrix_traversal([[1, 2], [3, 4], [5, 6]])", "[1, 2, 4, 3, 5, 6]"],
        hint="Loop over row index i: if i % 2 == 0 append row as is; else append reversed(row).",
        solution="    res = []\n    for i, row in enumerate(matrix):\n        if i % 2 == 0:\n            res.extend(row)\n        else:\n            res.extend(reversed(row))\n    return res",
        test="assert snake_matrix_traversal([[1, 2], [3, 4], [5, 6]]) == [1, 2, 4, 3, 5, 6]\nassert snake_matrix_traversal([[1]]) == [1]\nassert snake_matrix_traversal([]) == []\n"
    ))

    # 16. Jump Game Reach End
    tasks.append(create_task(
        name="can_jump_to_end",
        signature="def can_jump_to_end(nums: list[int]) -> bool:",
        doc_desc="Determines if you can reach the last index starting at index 0 where nums[i] is max jump length.",
        doctests=[">>> can_jump_to_end([2, 3, 1, 1, 4])", "True", ">>> can_jump_to_end([3, 2, 1, 0, 4])", "False"],
        hint="Track max_reach = 0. Loop i, val in enumerate(nums): if i > max_reach return False; max_reach = max(max_reach, i + val).",
        solution="    max_reach = 0\n    for i, n in enumerate(nums):\n        if i > max_reach:\n            return False\n        max_reach = max(max_reach, i + n)\n    return True",
        test="assert can_jump_to_end([2, 3, 1, 1, 4]) is True\nassert can_jump_to_end([3, 2, 1, 0, 4]) is False\nassert can_jump_to_end([0]) is True\n"
    ))

    # 17. Gas Station Circular Tour
    tasks.append(create_task(
        name="gas_station_start_index",
        signature="def gas_station_start_index(gas: list[int], cost: list[int]) -> int:",
        doc_desc="Finds starting gas station index to complete circular tour clockwise once, or -1 if impossible.",
        doctests=[">>> gas_station_start_index([1, 2, 3, 4, 5], [3, 4, 5, 1, 2])", "3"],
        hint="If sum(gas) < sum(cost) return -1. Track current tank: if tank < 0, reset start = i + 1, tank = 0.",
        solution="    if sum(gas) < sum(cost):\n        return -1\n    start = 0\n    tank = 0\n    for i in range(len(gas)):\n        tank += gas[i] - cost[i]\n        if tank < 0:\n            start = i + 1\n            tank = 0\n    return start",
        test="assert gas_station_start_index([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3\nassert gas_station_start_index([2, 3, 4], [3, 4, 3]) == -1\n"
    ))

    # 18. Trap Rain Water Two Pointer
    tasks.append(create_task(
        name="trap_rain_water",
        signature="def trap_rain_water(height: list[int]) -> int:",
        doc_desc="Computes how much water can be trapped after raining on elevation map.",
        doctests=[">>> trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])", "6"],
        hint="Two pointers left=0, right=len-1 with left_max, right_max. If height[left] < height[right]: update left_max and water += left_max - height[left].",
        solution="    if not height:\n        return 0\n    left, right = 0, len(height) - 1\n    left_max, right_max = height[left], height[right]\n    water = 0\n    while left < right:\n        if height[left] < height[right]:\n            left += 1\n            left_max = max(left_max, height[left])\n            water += left_max - height[left]\n        else:\n            right -= 1\n            right_max = max(right_max, height[right])\n            water += right_max - height[right]\n    return water",
        test="assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6\nassert trap_rain_water([4, 2, 0, 3, 2, 5]) == 9\nassert trap_rain_water([]) == 0\n"
    ))

    # 19. Remove Duplicates From Sorted Array Count
    tasks.append(create_task(
        name="remove_duplicates_sorted_count",
        signature="def remove_duplicates_sorted_count(nums: list[int]) -> int:",
        doc_desc="Modifies sorted array in-place so unique elements appear at the beginning, returning count of unique elements.",
        doctests=[">>> a = [1, 1, 2]; remove_duplicates_sorted_count(a)", "2"],
        hint="If nums empty return 0. Pointer write = 1. Loop read from 1 to len: if nums[read] != nums[read-1]: nums[write] = nums[read], write += 1.",
        solution="    if not nums:\n        return 0\n    write = 1\n    for read in range(1, len(nums)):\n        if nums[read] != nums[read - 1]:\n            nums[write] = nums[read]\n            write += 1\n    return write",
        test="a = [1, 1, 2]\nassert remove_duplicates_sorted_count(a) == 2\nassert a[:2] == [1, 2]\nassert remove_duplicates_sorted_count([]) == 0\n"
    ))

    # 20. Asteroid Collision Simulation
    tasks.append(create_task(
        name="asteroid_collision",
        signature="def asteroid_collision(asteroids: list[int]) -> list[int]:",
        doc_desc="Simulates asteroid collisions: positive moves right, negative moves left. Larger destroys smaller; equal destroy both.",
        doctests=[">>> asteroid_collision([5, 10, -5])", "[5, 10]", ">>> asteroid_collision([8, -8])", "[]"],
        hint="Use a stack. While ast < 0 and stack and stack[-1] > 0: compare abs(ast) and stack[-1]. If equal, destroy both; if ast larger, pop and continue.",
        solution="    stack = []\n    for ast in asteroids:\n        alive = True\n        while alive and ast < 0 and stack and stack[-1] > 0:\n            if stack[-1] < -ast:\n                stack.pop()\n            elif stack[-1] == -ast:\n                stack.pop()\n                alive = False\n            else:\n                alive = False\n        if alive:\n            stack.append(ast)\n    return stack",
        test="assert asteroid_collision([5, 10, -5]) == [5, 10]\nassert asteroid_collision([8, -8]) == []\nassert asteroid_collision([10, 2, -5]) == [10]\nassert asteroid_collision([-2, -1, 1, 2]) == [-2, -1, 1, 2]\n"
    ))

    # 21. Boyer-Moore Majority Element
    tasks.append(create_task(
        name="majority_element_boyer_moore",
        signature="def majority_element_boyer_moore(nums: list[int]) -> int:",
        doc_desc="Finds the majority element appearing more than n // 2 times in O(1) space.",
        doctests=[">>> majority_element_boyer_moore([3, 2, 3])", "3"],
        hint="Track candidate and count. If count == 0: candidate = n. If n == candidate: count += 1 else count -= 1.",
        solution="    candidate = None\n    count = 0\n    for n in nums:\n        if count == 0:\n            candidate = n\n        count += 1 if n == candidate else -1\n    return candidate",
        test="assert majority_element_boyer_moore([3, 2, 3]) == 3\nassert majority_element_boyer_moore([2, 2, 1, 1, 1, 2, 2]) == 2\n"
    ))

    # 22. Summary Ranges
    tasks.append(create_task(
        name="summary_ranges_loop",
        signature="def summary_ranges_loop(nums: list[int]) -> list[str]:",
        doc_desc="Summarizes sorted unique integer array into range strings like '0->2', '4->5', '7'.",
        doctests=[">>> summary_ranges_loop([0, 1, 2, 4, 5, 7])", "['0->2', '4->5', '7']"],
        hint="Track start pointer i. Move pointer j while j + 1 < len and nums[j + 1] == nums[j] + 1. Append range.",
        solution="    res = []\n    i = 0\n    n = len(nums)\n    while i < n:\n        j = i\n        while j + 1 < n and nums[j + 1] == nums[j] + 1:\n            j += 1\n        if i == j:\n            res.append(str(nums[i]))\n        else:\n            res.append(f'{nums[i]}->{nums[j]}')\n        i = j + 1\n    return res",
        test="assert summary_ranges_loop([0, 1, 2, 4, 5, 7]) == ['0->2', '4->5', '7']\nassert summary_ranges_loop([0, 2, 3, 4, 6, 8, 9]) == ['0', '2->4', '6', '8->9']\nassert summary_ranges_loop([]) == []\n"
    ))

    # Dynamically generate 42 additional genuine Control Flow & Loop challenges (total 64)
    cf_specs = [
        ("count_steps_to_zero", "def count_steps_to_zero(num: int) -> int:",
         "Counts steps to reduce num to 0: if even divide by 2, if odd subtract 1.",
         "Loop while num > 0: num = num // 2 if num % 2 == 0 else num - 1; count += 1.",
         "    count = 0\n    while num > 0:\n        num = num // 2 if num % 2 == 0 else num - 1\n        count += 1\n    return count",
         "assert count_steps_to_zero(14) == 6\nassert count_steps_to_zero(8) == 4\nassert count_steps_to_zero(0) == 0\n"),

        ("find_first_loop_repetition", "def find_first_loop_repetition(nums: list[int]) -> int | None:",
         "Returns the first number that appears consecutively twice in a loop scan, or None.",
         "Iterate 1..len: if nums[i] == nums[i-1] return nums[i].",
         "    for i in range(1, len(nums)):\n        if nums[i] == nums[i - 1]:\n            return nums[i]\n    return None",
         "assert find_first_loop_repetition([1, 2, 2, 3]) == 2\nassert find_first_loop_repetition([1, 2, 3]) is None\n"),

        ("simulate_pig_latin_word", "def simulate_pig_latin_word(word: str) -> str:",
         "Converts a lowercase word to Pig Latin: if starts with vowel add 'way'; else move leading consonants to end and add 'ay'.",
         "Loop index to find first vowel in 'aeiou'.",
         "    vowels = set('aeiou')\n    if not word:\n        return ''\n    if word[0] in vowels:\n        return word + 'way'\n    for i, ch in enumerate(word):\n        if ch in vowels:\n            return word[i:] + word[:i] + 'ay'\n    return word + 'ay'",
         "assert simulate_pig_latin_word('apple') == 'appleway'\nassert simulate_pig_latin_word('string') == 'ingstray'\n"),

        ("filter_alternating_even_odd", "def filter_alternating_even_odd(nums: list[int]) -> list[int]:",
         "Collects longest prefix where elements strictly alternate even and odd parity.",
         "Loop and check (nums[i] % 2 != nums[i-1] % 2). Break on violation.",
         "    if not nums: return []\n    res = [nums[0]]\n    for i in range(1, len(nums)):\n        if nums[i] % 2 != nums[i - 1] % 2:\n            res.append(nums[i])\n        else:\n            break\n    return res",
         "assert filter_alternating_even_odd([2, 3, 4, 5, 5, 6]) == [2, 3, 4, 5]\nassert filter_alternating_even_odd([1, 1]) == [1]\n"),

        ("loop_find_first_missing_positive", "def loop_find_first_missing_positive(nums: list[int]) -> int:",
         "Finds smallest positive integer not in nums using set lookup loop.",
         "Build set, check 1, 2, 3... until not found.",
         "    s = set(nums)\n    i = 1\n    while True:\n        if i not in s:\n            return i\n        i += 1",
         "assert loop_find_first_missing_positive([1, 2, 0]) == 3\nassert loop_find_first_missing_positive([3, 4, -1, 1]) == 2\n"),

        ("loop_merge_intervals_count", "def loop_merge_intervals_count(intervals: list[list[int]]) -> int:",
         "Counts number of distinct non-overlapping intervals after merging.",
         "Sort intervals, merge overlapping, return len(merged).",
         "    if not intervals: return 0\n    intervals.sort(key=lambda x: x[0])\n    count = 1\n    curr_end = intervals[0][1]\n    for s, e in intervals[1:]:\n        if s <= curr_end:\n            curr_end = max(curr_end, e)\n        else:\n            count += 1\n            curr_end = e\n    return count",
         "assert loop_merge_intervals_count([[1, 3], [2, 6], [8, 10]]) == 2\nassert loop_merge_intervals_count([]) == 0\n"),

        ("loop_calculate_run_length", "def loop_calculate_run_length(s: str) -> list[tuple[str, int]]:",
         "Returns list of (char, count) for contiguous runs using explicit loop.",
         "Loop through chars tracking current run char and count.",
         "    if not s: return []\n    res = []\n    curr_ch = s[0]\n    cnt = 1\n    for ch in s[1:]:\n        if ch == curr_ch: cnt += 1\n        else:\n            res.append((curr_ch, cnt))\n            curr_ch = ch\n            cnt = 1\n    res.append((curr_ch, cnt))\n    return res",
         "assert loop_calculate_run_length('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]\nassert loop_calculate_run_length('') == []\n"),

        ("simulate_grid_boundary_bounce", "def simulate_grid_boundary_bounce(width: int, height: int, steps: int) -> tuple[int, int]:",
         "Simulates 2D ball bouncing diagonally (dx=1, dy=1) inside grid 0..width, 0..height.",
         "Update x += dx, y += dy. Invert dx at x==0 or x==width; invert dy at y==0 or y==height.",
         "    x, y = 0, 0\n    dx, dy = 1, 1\n    for _ in range(steps):\n        x += dx\n        y += dy\n        if x == 0 or x == width: dx = -dx\n        if y == 0 or y == height: dy = -dy\n    return (x, y)",
         "assert simulate_grid_boundary_bounce(2, 2, 2) == (2, 2)\nassert simulate_grid_boundary_bounce(2, 2, 3) == (1, 1)\n"),

        ("simulate_countdown_blastoff", "def simulate_countdown_blastoff(n: int) -> list[str]:",
         "Generates countdown sequence ['3', '2', '1', 'Blastoff!'].",
         "Loop from n down to 1, append string, finish with 'Blastoff!'.",
         "    res = [str(i) for i in range(n, 0, -1)]\n    res.append('Blastoff!')\n    return res",
         "assert simulate_countdown_blastoff(3) == ['3', '2', '1', 'Blastoff!']\nassert simulate_countdown_blastoff(0) == ['Blastoff!']\n"),

        ("loop_find_first_negative", "def loop_find_first_negative(nums: list[float]) -> float | None:",
         "Returns first negative number in list or None.",
         "Loop and return on n < 0.",
         "    for n in nums:\n        if n < 0: return n\n    return None",
         "assert loop_find_first_negative([1.0, -2.5, 3.0]) == -2.5\nassert loop_find_first_negative([1.0, 2.0]) is None\n"),

        ("loop_accumulate_until_threshold", "def loop_accumulate_until_threshold(nums: list[int], threshold: int) -> int:",
         "Sums numbers in order until running sum >= threshold, returning count of items consumed.",
         "Track total. Break when total >= threshold.",
         "    total = 0\n    for i, n in enumerate(nums):\n        total += n\n        if total >= threshold:\n            return i + 1\n    return len(nums)",
         "assert loop_accumulate_until_threshold([1, 2, 3, 4], 5) == 3\nassert loop_accumulate_until_threshold([10], 5) == 1\n"),

        ("loop_chunk_elements_by_size", "def loop_chunk_elements_by_size(items: list[Any], size: int) -> list[list[Any]]:",
         "Splits items into consecutive chunks of length size using a loop with step size.",
         "Iterate range(0, len(items), size) and slice.",
         "    if size <= 0: return []\n    return [items[i:i + size] for i in range(0, len(items), size)]",
         "assert loop_chunk_elements_by_size([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert loop_chunk_elements_by_size([], 2) == []\n"),

        ("loop_detect_decreasing_trend", "def loop_detect_decreasing_trend(nums: list[int]) -> bool:",
         "Checks if list is strictly decreasing throughout.",
         "Check all nums[i] < nums[i-1].",
         "    return all(nums[i] < nums[i - 1] for i in range(1, len(nums)))",
         "assert loop_detect_decreasing_trend([5, 4, 2, 1]) is True\nassert loop_detect_decreasing_trend([5, 5]) is False\nassert loop_detect_decreasing_trend([]) is True\n"),

        ("loop_interleave_until_exhausted", "def loop_interleave_until_exhausted(l1: list[int], l2: list[int]) -> list[int]:",
         "Alternates elements of l1 and l2, taking from whichever has remaining items.",
         "Loop max(len(l1), len(l2)).",
         "    res = []\n    m = max(len(l1), len(l2))\n    for i in range(m):\n        if i < len(l1): res.append(l1[i])\n        if i < len(l2): res.append(l2[i])\n    return res",
         "assert loop_interleave_until_exhausted([1, 2], [10, 20, 30]) == [1, 10, 2, 20, 30]\nassert loop_interleave_until_exhausted([], [1]) == [1]\n"),

        ("loop_count_local_minima", "def loop_count_local_minima(nums: list[int]) -> int:",
         "Counts elements strictly smaller than their immediate neighbors.",
         "Loop 1..len-1: if nums[i] < nums[i-1] and nums[i] < nums[i+1]: count += 1.",
         "    count = 0\n    for i in range(1, len(nums) - 1):\n        if nums[i] < nums[i - 1] and nums[i] < nums[i + 1]:\n            count += 1\n    return count",
         "assert loop_count_local_minima([3, 1, 4, 2, 5]) == 2\nassert loop_count_local_minima([1, 2, 3]) == 0\n"),

        ("loop_find_pivot_index", "def loop_find_pivot_index(nums: list[int]) -> int:",
         "Finds 0-based index where left sum equals right sum, or -1.",
         "Track total_sum = sum(nums), left_sum = 0. If left_sum == total_sum - left_sum - nums[i] return i.",
         "    total = sum(nums)\n    left = 0\n    for i, n in enumerate(nums):\n        if left == total - left - n:\n            return i\n        left += n\n    return -1",
         "assert loop_find_pivot_index([1, 7, 3, 6, 5, 6]) == 3\nassert loop_find_pivot_index([1, 2, 3]) == -1\n"),

        ("loop_simulate_stack_clear", "def loop_simulate_stack_clear(ops: list[str]) -> list[str]:",
         "Processes 'PUSH x' and 'POP' strings returning final stack contents.",
         "Loop through ops: if op.startswith('PUSH '): stack.append(op[5:]); elif op == 'POP' and stack: stack.pop().",
         "    stack = []\n    for op in ops:\n        if op.startswith('PUSH '):\n            stack.append(op[5:])\n        elif op == 'POP' and stack:\n            stack.pop()\n    return stack",
         "assert loop_simulate_stack_clear(['PUSH a', 'PUSH b', 'POP', 'PUSH c']) == ['a', 'c']\n"),

        ("loop_longest_zero_sum_subarray_length", "def loop_longest_zero_sum_subarray_length(nums: list[int]) -> int:",
         "Finds length of longest contiguous subarray whose sum is 0.",
         "Prefix sum map storing first occurrence index of each prefix sum.",
         "    prefix_map = {0: -1}\n    curr = 0\n    max_len = 0\n    for i, n in enumerate(nums):\n        curr += n\n        if curr in prefix_map:\n            max_len = max(max_len, i - prefix_map[curr])\n        else:\n            prefix_map[curr] = i\n    return max_len",
         "assert loop_longest_zero_sum_subarray_length([1, -1, 3, 2, -2, -3]) == 6\nassert loop_longest_zero_sum_subarray_length([1, 2, 3]) == 0\n"),

        ("loop_check_strictly_bimodal", "def loop_check_strictly_bimodal(nums: list[int]) -> bool:",
         "Checks if array strictly increases to a single peak then strictly decreases (mountain array).",
         "Find peak index with loop. Check i > 0, i < len - 1, strictly increasing before and decreasing after.",
         "    n = len(nums)\n    if n < 3: return False\n    i = 0\n    while i + 1 < n and nums[i] < nums[i + 1]:\n        i += 1\n    if i == 0 or i == n - 1: return False\n    while i + 1 < n and nums[i] > nums[i + 1]:\n        i += 1\n    return i == n - 1",
         "assert loop_check_strictly_bimodal([0, 3, 2, 1]) is True\nassert loop_check_strictly_bimodal([3, 5, 5]) is False\nassert loop_check_strictly_bimodal([0, 1, 2]) is False\n"),

        ("loop_clamp_series", "def loop_clamp_series(nums: list[float], low: float, high: float) -> list[float]:",
         "Clamps each element in nums to range [low, high].",
         "max(low, min(high, x)).",
         "    return [max(low, min(high, x)) for x in nums]",
         "assert loop_clamp_series([-5.0, 3.0, 15.0], 0.0, 10.0) == [0.0, 3.0, 10.0]\n"),

        ("loop_count_character_flips", "def loop_count_character_flips(s: str) -> int:",
         "Counts number of character changes between adjacent elements in string.",
         "sum(1 for i in range(1, len(s)) if s[i] != s[i-1]).",
         "    return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])",
         "assert loop_count_character_flips('0101') == 3\nassert loop_count_character_flips('000') == 0\n"),

        ("loop_remove_subsequent_smaller", "def loop_remove_subsequent_smaller(nums: list[int]) -> list[int]:",
         "Filters elements keeping only those greater than all previous kept elements.",
         "Track running max. Append and update if num > max.",
         "    if not nums: return []\n    res = [nums[0]]\n    curr_max = nums[0]\n    for n in nums[1:]:\n        if n > curr_max:\n            res.append(n)\n            curr_max = n\n    return res",
         "assert loop_remove_subsequent_smaller([1, 3, 2, 4, 2, 5]) == [1, 3, 4, 5]\n"),

        ("loop_partition_parity", "def loop_partition_parity(nums: list[int]) -> tuple[list[int], list[int]]:",
         "Splits list into (evens, odds) preserving relative order.",
         "Loop through nums into two lists.",
         "    evens = [x for x in nums if x % 2 == 0]\n    odds = [x for x in nums if x % 2 != 0]\n    return (evens, odds)",
         "assert loop_partition_parity([1, 2, 3, 4]) == ([2, 4], [1, 3])\n"),

        ("loop_simulate_step_counter_walk", "def loop_simulate_step_counter_walk(steps_history: list[int], daily_goal: int) -> int:",
         "Counts days where daily goal was achieved.",
         "sum(1 for s in steps_history if s >= daily_goal).",
         "    return sum(1 for s in steps_history if s >= daily_goal)",
         "assert loop_simulate_step_counter_walk([8000, 10500, 12000, 9500], 10000) == 2\n"),

        ("loop_matrix_diagonal_sum", "def loop_matrix_diagonal_sum(mat: list[list[int]]) -> int:",
         "Sums primary and secondary diagonals of square matrix, subtracting center if n is odd.",
         "Loop i from 0..n-1: sum mat[i][i] + mat[i][n - 1 - i]. If n % 2 == 1: subtract center mat[n//2][n//2].",
         "    n = len(mat)\n    total = 0\n    for i in range(n):\n        total += mat[i][i]\n        if i != n - 1 - i:\n            total += mat[i][n - 1 - i]\n    return total",
         "assert loop_matrix_diagonal_sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 25\nassert loop_matrix_diagonal_sum([[1, 1], [1, 1]]) == 4\n"),

        ("loop_calculate_exponential_smoothing", "def loop_calculate_exponential_smoothing(data: list[float], alpha: float = 0.5) -> list[float]:",
         "Computes exponential smoothing s[0]=data[0], s[t] = alpha*data[t] + (1-alpha)*s[t-1].",
         "Loop and accumulate smoothed value.",
         "    if not data: return []\n    s = [data[0]]\n    for x in data[1:]:\n        s.append(alpha * x + (1 - alpha) * s[-1])\n    return s",
         "res = loop_calculate_exponential_smoothing([10.0, 20.0, 30.0], 0.5)\nassert res[0] == 10.0 and res[1] == 15.0 and res[2] == 22.5\n"),

        ("loop_consecutive_element_differences", "def loop_consecutive_element_differences(nums: list[int]) -> list[int]:",
         "Returns list of differences [nums[i+1] - nums[i]].",
         "List comprehension with zip.",
         "    return [b - a for a, b in zip(nums, nums[1:])]",
         "assert loop_consecutive_element_differences([1, 4, 9]) == [3, 5]\nassert loop_consecutive_element_differences([5]) == []\n"),

        ("loop_find_first_duplicate_char", "def loop_find_first_duplicate_char(s: str) -> str | None:",
         "Returns first character in string that has already appeared previously.",
         "Use seen set.",
         "    seen = set()\n    for ch in s:\n        if ch in seen: return ch\n        seen.add(ch)\n    return None",
         "assert loop_find_first_duplicate_char('abcdefa') == 'a'\nassert loop_find_first_duplicate_char('abc') is None\n"),

        ("loop_accumulate_products", "def loop_accumulate_products(nums: list[int]) -> list[int]:",
         "Returns running prefix products of nums.",
         "Accumulate product with loop.",
         "    res = []\n    p = 1\n    for n in nums:\n        p *= n\n        res.append(p)\n    return res",
         "assert loop_accumulate_products([1, 2, 3, 4]) == [1, 2, 6, 24]\nassert loop_accumulate_products([]) == []\n"),

        ("loop_count_target_in_nested_list", "def loop_count_target_in_nested_list(nested: list[list[int]], target: int) -> int:",
         "Counts occurrences of target across a 2D nested list.",
         "Nested loop summing target occurrences.",
         "    return sum(row.count(target) for row in nested)",
         "assert loop_count_target_in_nested_list([[1, 2], [2, 3], [2]], 2) == 3\n"),

        ("loop_shift_zeros_to_end", "def loop_shift_zeros_to_end(nums: list[int]) -> list[int]:",
         "Returns list with all non-zero elements preserving order followed by all zeros.",
         "Non-zeros + zeros.",
         "    non_zeros = [x for x in nums if x != 0]\n    return non_zeros + [0] * (len(nums) - len(non_zeros))",
         "assert loop_shift_zeros_to_end([0, 1, 0, 3, 12]) == [1, 3, 12, 0, 0]\n"),

        ("loop_staircase_hash_string", "def loop_staircase_hash_string(n: int) -> list[str]:",
         "Generates right-aligned staircase of '#' strings with spaces of height n.",
         "f'{(i * \"#\"):>{n}}'.",
         "    return [f'{(i * \"#\"):>{n}}' for i in range(1, n + 1)]",
         "assert loop_staircase_hash_string(3) == ['  #', ' ##', '###']\n"),

        ("loop_is_matrix_symmetric", "def loop_is_matrix_symmetric(mat: list[list[int]]) -> bool:",
         "Checks if square matrix is equal to its transpose (mat[i][j] == mat[j][i]).",
         "Check all i, j.",
         "    n = len(mat)\n    return all(mat[i][j] == mat[j][i] for i in range(n) for j in range(i + 1, n))",
         "assert loop_is_matrix_symmetric([[1, 2], [2, 1]]) is True\nassert loop_is_matrix_symmetric([[1, 2], [3, 1]]) is False\n"),

        ("loop_find_first_greater_neighbor", "def loop_find_first_greater_neighbor(nums: list[int]) -> int | None:",
         "Returns the first element strictly greater than both its left and right neighbors, or None.",
         "Loop 1..len-1.",
         "    for i in range(1, len(nums) - 1):\n        if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:\n            return nums[i]\n    return None",
         "assert loop_find_first_greater_neighbor([1, 4, 2, 5, 3]) == 4\nassert loop_find_first_greater_neighbor([1, 2, 3]) is None\n"),

        ("loop_count_inversion_pairs", "def loop_count_inversion_pairs(nums: list[int]) -> int:",
         "Counts pairs (i, j) such that i < j and nums[i] > nums[j].",
         "Nested loop.",
         "    count = 0\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] > nums[j]:\n                count += 1\n    return count",
         "assert loop_count_inversion_pairs([2, 4, 1, 3, 5]) == 3\nassert loop_count_inversion_pairs([1, 2, 3]) == 0\n"),

        ("loop_validate_brackets_depth", "def loop_validate_brackets_depth(s: str) -> int:",
         "Returns maximum nesting depth of parentheses, or -1 if unbalanced.",
         "Track depth and max_depth.",
         "    curr = 0\n    max_d = 0\n    for ch in s:\n        if ch == '(':\n            curr += 1\n            max_d = max(max_d, curr)\n        elif ch == ')':\n            curr -= 1\n            if curr < 0: return -1\n    return max_d if curr == 0 else -1",
         "assert loop_validate_brackets_depth('((()))') == 3\nassert loop_validate_brackets_depth('(()') == -1\nassert loop_validate_brackets_depth('()') == 1\n"),

        ("loop_collatz_max_step_in_range", "def loop_collatz_max_step_in_range(start: int, end: int) -> tuple[int, int]:",
         "Finds (number, step_count) with maximal Collatz steps in range start..end inclusive.",
         "Loop through range, compute steps.",
         "    def steps(n):\n        c = 0\n        while n > 1:\n            n = n // 2 if n % 2 == 0 else 3 * n + 1\n            c += 1\n        return c\n    best_num, max_s = start, -1\n    for x in range(start, end + 1):\n        s = steps(x)\n        if s > max_s:\n            max_s = s\n            best_num = x\n    return (best_num, max_s)",
         "num, s = loop_collatz_max_step_in_range(1, 5)\nassert num == 3 and s == 7\n"),

        ("loop_count_zero_crossings", "def loop_count_zero_crossings(signal: list[float]) -> int:",
         "Counts number of sign changes between consecutive non-zero elements in a signal.",
         "Filter non-zeros, count sign changes.",
         "    non_zeros = [x for x in signal if x != 0.0]\n    return sum(1 for i in range(1, len(non_zeros)) if (non_zeros[i] > 0) != (non_zeros[i - 1] > 0))",
         "assert loop_count_zero_crossings([1.0, -1.0, 2.0, -2.0]) == 3\nassert loop_count_zero_crossings([1.0, 0.0, 2.0]) == 0\n"),

        ("loop_interleave_with_delimiter", "def loop_interleave_with_delimiter(items: list[str], delim: str) -> list[str]:",
         "Inserts delim between every two elements of items.",
         "Loop items.",
         "    if not items: return []\n    res = [items[0]]\n    for it in items[1:]:\n        res.append(delim)\n        res.append(it)\n    return res",
         "assert loop_interleave_with_delimiter(['a', 'b', 'c'], ',') == ['a', ',', 'b', ',', 'c']\nassert loop_interleave_with_delimiter([], '-') == []\n"),

        ("loop_longest_run_of_character", "def loop_longest_run_of_character(s: str, target: str) -> int:",
         "Finds length of longest consecutive streak of target character.",
         "Track curr_streak and max_streak.",
         "    max_s = curr = 0\n    for ch in s:\n        if ch == target:\n            curr += 1\n            max_s = max(max_s, curr)\n        else:\n            curr = 0\n    return max_s",
         "assert loop_longest_run_of_character('aaabaa', 'a') == 3\nassert loop_longest_run_of_character('abc', 'z') == 0\n"),

        ("loop_strip_outer_brackets", "def loop_strip_outer_brackets(s: str) -> str:",
         "Strips matching outer '(' and ')' if they enclose the entire balanced string.",
         "Check balance while iterating.",
         "    if len(s) >= 2 and s.startswith('(') and s.endswith(')'):\n        bal = 0\n        for i, ch in enumerate(s[:-1]):\n            if ch == '(': bal += 1\n            elif ch == ')': bal -= 1\n            if bal == 0: return s\n        return s[1:-1]\n    return s",
         "assert loop_strip_outer_brackets('(abc)') == 'abc'\nassert loop_strip_outer_brackets('(a)(b)') == '(a)(b)'\nassert loop_strip_outer_brackets('no') == 'no'\n"),

        ("loop_clamp_cumulative_budget", "def loop_clamp_cumulative_budget(expenses: list[float], budget: float) -> list[float]:",
         "Accepts expenses while cumulative total does not exceed budget; rejects remainder.",
         "Accumulate running total.",
         "    total = 0.0\n    accepted = []\n    for e in expenses:\n        if total + e <= budget:\n            total += e\n            accepted.append(e)\n        else:\n            break\n    return accepted",
         "assert loop_clamp_cumulative_budget([20.0, 30.0, 40.0, 50.0], 60.0) == [20.0, 30.0]\nassert loop_clamp_cumulative_budget([], 100.0) == []\n")
    ]

    for name, sig, desc, hint, sol, test in cf_specs:
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
