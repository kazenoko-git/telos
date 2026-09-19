"""
Category 4: Object-Oriented Programming (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_oop_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Vector2D
    tasks.append(create_task(
        name="Vector2D",
        signature="class Vector2D:\n    def __init__(self, x: float, y: float):\n        self.x = float(x)\n        self.y = float(y)",
        doc_desc="2D Vector supporting addition (+), subtraction (-), dot product (.dot()), equality (==), and magnitude.",
        doctests=[">>> v1 = Vector2D(1, 2); v2 = Vector2D(3, 4); (v1 + v2).x", "4.0"],
        hint="Implement __add__, __sub__, dot(other), magnitude(), and __eq__ comparing abs(x - other.x) < 1e-6.",
        solution="    def __add__(self, other: 'Vector2D') -> 'Vector2D':\n        return Vector2D(self.x + other.x, self.y + other.y)\n    def __sub__(self, other: 'Vector2D') -> 'Vector2D':\n        return Vector2D(self.x - other.x, self.y - other.y)\n    def dot(self, other: 'Vector2D') -> float:\n        return self.x * other.x + self.y * other.y\n    def magnitude(self) -> float:\n        import math\n        return math.hypot(self.x, self.y)\n    def __eq__(self, other: Any) -> bool:\n        if not isinstance(other, Vector2D):\n            return False\n        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6",
        test="v1 = Vector2D(1, 2)\nv2 = Vector2D(3, 4)\nassert v1 + v2 == Vector2D(4, 6)\nassert v2 - v1 == Vector2D(2, 2)\nassert v1.dot(v2) == 11.0\nassert abs(Vector2D(3, 4).magnitude() - 5.0) < 1e-6\n"
    ))

    # 2. Temperature
    tasks.append(create_task(
        name="Temperature",
        signature="class Temperature:\n    def __init__(self, celsius: float):\n        self.celsius = float(celsius)",
        doc_desc="Stores temperature in Celsius with property accessors for fahrenheit and kelvin, and from_fahrenheit classmethod.",
        doctests=[">>> t = Temperature(0); abs(t.fahrenheit - 32.0) < 1e-5", "True"],
        hint="Use @property for fahrenheit (c * 9/5 + 32) and kelvin (c + 273.15). Implement @classmethod from_fahrenheit(cls, f).",
        solution="    @property\n    def fahrenheit(self) -> float:\n        return self.celsius * 9.0 / 5.0 + 32.0\n    @property\n    def kelvin(self) -> float:\n        return self.celsius + 273.15\n    @classmethod\n    def from_fahrenheit(cls, f: float) -> 'Temperature':\n        return cls((f - 32.0) * 5.0 / 9.0)",
        test="t = Temperature(0)\nassert abs(t.fahrenheit - 32.0) < 1e-5\nassert abs(t.kelvin - 273.15) < 1e-5\nt_f = Temperature.from_fahrenheit(212.0)\nassert abs(t_f.celsius - 100.0) < 1e-5\n"
    ))

    # 3. BankAccount
    tasks.append(create_task(
        name="BankAccount",
        signature="class BankAccount:\n    def __init__(self, initial_balance: float = 0.0):\n        self.balance = float(initial_balance)\n        self.history = [('INIT', float(initial_balance))]",
        doc_desc="Represents a bank account with deposit, withdraw, and transaction history tracking.",
        doctests=[">>> acc = BankAccount(100); acc.deposit(50); acc.balance", "150.0"],
        hint="deposit(amount): add amount and record ('DEPOSIT', amount). withdraw(amount): if amount <= balance deduct and record ('WITHDRAW', amount) and return True; else return False.",
        solution="    def deposit(self, amount: float) -> None:\n        if amount <= 0:\n            raise ValueError('Deposit must be positive')\n        self.balance += amount\n        self.history.append(('DEPOSIT', float(amount)))\n    def withdraw(self, amount: float) -> bool:\n        if amount <= 0 or amount > self.balance:\n            return False\n        self.balance -= amount\n        self.history.append(('WITHDRAW', float(amount)))\n        return True\n    def get_history(self) -> list[tuple[str, float]]:\n        return list(self.history)",
        test="acc = BankAccount(100)\nacc.deposit(50)\nassert acc.balance == 150.0\nassert acc.withdraw(30) is True\nassert acc.balance == 120.0\nassert acc.withdraw(200) is False\nassert len(acc.get_history()) == 3\n"
    ))

    # 4. ShoppingCart
    tasks.append(create_task(
        name="ShoppingCart",
        signature="class ShoppingCart:\n    def __init__(self):\n        self.items = {}  # item_name -> {'price': float, 'qty': int}",
        doc_desc="Shopping cart supporting add_item, remove_item, total_price, and apply_discount(percent).",
        doctests=[">>> sc = ShoppingCart(); sc.add_item('apple', 1.5, 2); sc.total_price()", "3.0"],
        hint="Store items in dict. add_item updates qty and price. remove_item decrements qty (deletes if 0). total_price sums price * qty.",
        solution="    def add_item(self, name: str, price: float, qty: int = 1) -> None:\n        if name in self.items:\n            self.items[name]['qty'] += qty\n            self.items[name]['price'] = price\n        else:\n            self.items[name] = {'price': float(price), 'qty': qty}\n    def remove_item(self, name: str, qty: int = 1) -> bool:\n        if name not in self.items:\n            return False\n        if self.items[name]['qty'] <= qty:\n            del self.items[name]\n        else:\n            self.items[name]['qty'] -= qty\n        return True\n    def total_price(self, discount_pct: float = 0.0) -> float:\n        subtotal = sum(v['price'] * v['qty'] for v in self.items.values())\n        return subtotal * (1.0 - discount_pct / 100.0)",
        test="sc = ShoppingCart()\nsc.add_item('apple', 1.5, 2)\nassert sc.total_price() == 3.0\nsc.add_item('banana', 2.0, 1)\nassert sc.total_price(discount_pct=10.0) == 4.5\nassert sc.remove_item('apple', 1) is True\nassert sc.items['apple']['qty'] == 1\n"
    ))

    # 5. StackClass
    tasks.append(create_task(
        name="StackClass",
        signature="class StackClass:\n    def __init__(self):\n        self.data = []",
        doc_desc="LIFO Stack with push, pop, peek, is_empty, and size methods.",
        doctests=[">>> s = StackClass(); s.push(10); s.peek()", "10"],
        hint="Use list operations: append for push, pop for pop (returns None if empty), [-1] for peek.",
        solution="    def push(self, val: Any) -> None:\n        self.data.append(val)\n    def pop(self) -> Any | None:\n        return self.data.pop() if self.data else None\n    def peek(self) -> Any | None:\n        return self.data[-1] if self.data else None\n    def is_empty(self) -> bool:\n        return len(self.data) == 0\n    def size(self) -> int:\n        return len(self.data)",
        test="s = StackClass()\nassert s.is_empty() is True\ns.push(1)\ns.push(2)\nassert s.peek() == 2\nassert s.pop() == 2\nassert s.size() == 1\n"
    ))

    # 6. QueueClass
    tasks.append(create_task(
        name="QueueClass",
        signature="class QueueClass:\n    def __init__(self):\n        from collections import deque\n        self.data = deque()",
        doc_desc="FIFO Queue with enqueue, dequeue, peek, is_empty, and size methods.",
        doctests=[">>> q = QueueClass(); q.enqueue('a'); q.enqueue('b'); q.dequeue()", "'a'"],
        hint="Use collections.deque: append for enqueue, popleft for dequeue (returns None if empty).",
        solution="    def enqueue(self, val: Any) -> None:\n        self.data.append(val)\n    def dequeue(self) -> Any | None:\n        return self.data.popleft() if self.data else None\n    def peek(self) -> Any | None:\n        return self.data[0] if self.data else None\n    def is_empty(self) -> bool:\n        return len(self.data) == 0\n    def size(self) -> int:\n        return len(self.data)",
        test="q = QueueClass()\nq.enqueue(1)\nq.enqueue(2)\nassert q.peek() == 1\nassert q.dequeue() == 1\nassert q.size() == 1\nassert q.dequeue() == 2\nassert q.is_empty() is True\n"
    ))

    # 7. PriorityQueueClass
    tasks.append(create_task(
        name="PriorityQueueClass",
        signature="class PriorityQueueClass:\n    def __init__(self):\n        self.heap = []",
        doc_desc="Min-Priority Queue supporting push(item, priority), pop(), peek(), is_empty, and size.",
        doctests=[">>> pq = PriorityQueueClass(); pq.push('x', 2); pq.push('y', 1); pq.pop()", "'y'"],
        hint="Use heapq module with (priority, item) tuples. pop() returns item associated with min priority.",
        solution="    def push(self, item: Any, priority: float) -> None:\n        import heapq\n        heapq.heappush(self.heap, (priority, item))\n    def pop(self) -> Any | None:\n        import heapq\n        return heapq.heappop(self.heap)[1] if self.heap else None\n    def peek(self) -> Any | None:\n        return self.heap[0][1] if self.heap else None\n    def is_empty(self) -> bool:\n        return len(self.heap) == 0\n    def size(self) -> int:\n        return len(self.heap)",
        test="pq = PriorityQueueClass()\npq.push('low', 10)\npq.push('high', 1)\nassert pq.peek() == 'high'\nassert pq.pop() == 'high'\nassert pq.size() == 1\nassert pq.pop() == 'low'\nassert pq.is_empty() is True\n"
    ))

    # 8. BoundedCounter
    tasks.append(create_task(
        name="BoundedCounter",
        signature="class BoundedCounter:\n    def __init__(self, min_val: int = 0, max_val: int = 10, start_val: int | None = None):\n        self.min_val = min_val\n        self.max_val = max_val\n        self.value = start_val if start_val is not None else min_val",
        doc_desc="A counter clamped strictly between min_val and max_val with increment, decrement, and reset.",
        doctests=[">>> c = BoundedCounter(0, 5, 4); c.increment(); c.increment(); c.value", "5"],
        hint="increment(step=1): set value = min(max_val, value + step). decrement(step=1): set value = max(min_val, value - step). reset(): set to min_val.",
        solution="    def increment(self, step: int = 1) -> int:\n        self.value = min(self.max_val, self.value + step)\n        return self.value\n    def decrement(self, step: int = 1) -> int:\n        self.value = max(self.min_val, self.value - step)\n        return self.value\n    def reset(self) -> None:\n        self.value = self.min_val",
        test="c = BoundedCounter(0, 5, 4)\nassert c.increment() == 5\nassert c.increment() == 5\nassert c.decrement(2) == 3\nc.reset()\nassert c.value == 0\n"
    ))

    # 9. TimerClock
    tasks.append(create_task(
        name="TimerClock",
        signature="class TimerClock:\n    def __init__(self, hours: int = 0, minutes: int = 0, seconds: int = 0):\n        self.total_secs = hours * 3600 + minutes * 60 + seconds",
        doc_desc="Clock tracking time with tick(seconds), reset(), and string formatting 'HH:MM:SS'.",
        doctests=[">>> c = TimerClock(0, 1, 30); c.tick(30); str(c)", "'00:02:00'"],
        hint="Maintain total seconds. tick(n) adds n seconds. Implement __str__ formatting hours:02d:minutes:02d:seconds:02d.",
        solution="    def tick(self, s: int = 1) -> None:\n        self.total_secs += s\n    def reset(self) -> None:\n        self.total_secs = 0\n    def __str__(self) -> str:\n        h = (self.total_secs // 3600) % 24\n        m = (self.total_secs % 3600) // 60\n        s = self.total_secs % 60\n        return f'{h:02d}:{m:02d}:{s:02d}'",
        test="c = TimerClock(0, 1, 30)\nc.tick(30)\nassert str(c) == '00:02:00'\nc.tick(3600)\nassert str(c) == '01:02:00'\nc.reset()\nassert str(c) == '00:00:00'\n"
    ))

    # 10. Point3D
    tasks.append(create_task(
        name="Point3D",
        signature="class Point3D:\n    def __init__(self, x: float, y: float, z: float):\n        self.x = float(x)\n        self.y = float(y)\n        self.z = float(z)",
        doc_desc="Represents a 3D point with distance_to(other), translate(dx, dy, dz), and equality.",
        doctests=[">>> p1 = Point3D(0, 0, 0); p2 = Point3D(1, 2, 2); p1.distance_to(p2)", "3.0"],
        hint="distance_to: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2). translate returns a new Point3D.",
        solution="    def distance_to(self, other: 'Point3D') -> float:\n        import math\n        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)\n    def translate(self, dx: float, dy: float, dz: float) -> 'Point3D':\n        return Point3D(self.x + dx, self.y + dy, self.z + dz)\n    def __eq__(self, other: Any) -> bool:\n        if not isinstance(other, Point3D):\n            return False\n        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6 and abs(self.z - other.z) < 1e-6",
        test="p1 = Point3D(0, 0, 0)\np2 = Point3D(1, 2, 2)\nassert abs(p1.distance_to(p2) - 3.0) < 1e-6\nassert p1.translate(1, 2, 2) == p2\n"
    ))

    # 11. Rectangle
    tasks.append(create_task(
        name="Rectangle",
        signature="class Rectangle:\n    def __init__(self, width: float, height: float):\n        self.width = float(width)\n        self.height = float(height)",
        doc_desc="2D Rectangle with area(), perimeter(), is_square(), and scale(factor).",
        doctests=[">>> r = Rectangle(4, 5); r.area()", "20.0"],
        hint="area: w * h, perimeter: 2 * (w + h), is_square: abs(w - h) < 1e-6, scale(factor): returns new Rectangle(w * factor, h * factor).",
        solution="    def area(self) -> float:\n        return self.width * self.height\n    def perimeter(self) -> float:\n        return 2.0 * (self.width + self.height)\n    def is_square(self) -> bool:\n        return abs(self.width - self.height) < 1e-6\n    def scale(self, factor: float) -> 'Rectangle':\n        return Rectangle(self.width * factor, self.height * factor)",
        test="r = Rectangle(4, 5)\nassert r.area() == 20.0\nassert r.perimeter() == 18.0\nassert r.is_square() is False\nr_sq = Rectangle(4, 4)\nassert r_sq.is_square() is True\nassert r.scale(2).width == 8.0\n"
    ))

    # 12. Circle
    tasks.append(create_task(
        name="Circle",
        signature="class Circle:\n    def __init__(self, radius: float):\n        self.radius = float(radius)",
        doc_desc="2D Circle with diameter(), area(), circumference(), and contains_point(px, py, cx=0, cy=0).",
        doctests=[">>> c = Circle(3); c.diameter()", "6.0"],
        hint="Use math.pi. diameter = 2 * r, area = pi * r^2, circumference = 2 * pi * r, contains_point checks (px-cx)^2 + (py-cy)^2 <= r^2.",
        solution="    def diameter(self) -> float:\n        return 2.0 * self.radius\n    def area(self) -> float:\n        import math\n        return math.pi * self.radius ** 2\n    def circumference(self) -> float:\n        import math\n        return 2.0 * math.pi * self.radius\n    def contains_point(self, px: float, py: float, cx: float = 0.0, cy: float = 0.0) -> bool:\n        return (px - cx)**2 + (py - cy)**2 <= self.radius**2 + 1e-9",
        test="import math\nc = Circle(3)\nassert c.diameter() == 6.0\nassert abs(c.area() - 9.0 * math.pi) < 1e-5\nassert abs(c.circumference() - 6.0 * math.pi) < 1e-5\nassert c.contains_point(1, 1) is True\nassert c.contains_point(4, 4) is False\n"
    ))

    # 13. FractionClass
    tasks.append(create_task(
        name="FractionClass",
        signature="class FractionClass:\n    def __init__(self, num: int, den: int = 1):\n        import math\n        if den == 0: raise ZeroDivisionError('Denominator cannot be zero')\n        g = math.gcd(abs(num), abs(den))\n        sign = -1 if (num < 0) ^ (den < 0) else 1\n        self.num = sign * (abs(num) // g)\n        self.den = abs(den) // g",
        doc_desc="Represents an irreducible fraction with __add__, __mul__, __float__, and __eq__.",
        doctests=[">>> f1 = FractionClass(1, 2); f2 = FractionClass(1, 3); (f1 + f2).num", "5"],
        hint="Simplify using gcd during init. __add__: (n1*d2 + n2*d1)/(d1*d2). __mul__: (n1*n2)/(d1*d2). __float__: num / den.",
        solution="    def __add__(self, other: 'FractionClass') -> 'FractionClass':\n        return FractionClass(self.num * other.den + other.num * self.den, self.den * other.den)\n    def __mul__(self, other: 'FractionClass') -> 'FractionClass':\n        return FractionClass(self.num * other.num, self.den * other.den)\n    def __float__(self) -> float:\n        return self.num / self.den\n    def __eq__(self, other: Any) -> bool:\n        if not isinstance(other, FractionClass): return False\n        return self.num == other.num and self.den == other.den",
        test="f1 = FractionClass(1, 2)\nf2 = FractionClass(1, 3)\nassert f1 + f2 == FractionClass(5, 6)\nassert f1 * f2 == FractionClass(1, 6)\nassert float(FractionClass(3, 4)) == 0.75\nassert FractionClass(2, 4) == FractionClass(1, 2)\n"
    ))

    # 14. ComplexNumber
    tasks.append(create_task(
        name="ComplexNumber",
        signature="class ComplexNumber:\n    def __init__(self, real: float, imag: float = 0.0):\n        self.real = float(real)\n        self.imag = float(imag)",
        doc_desc="Complex number with addition, multiplication, magnitude (__abs__), and equality.",
        doctests=[">>> c1 = ComplexNumber(1, 2); c2 = ComplexNumber(3, 4); (c1 + c2).real", "4.0"],
        hint="Multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i. Magnitude: sqrt(real^2 + imag^2).",
        solution="    def __add__(self, other: 'ComplexNumber') -> 'ComplexNumber':\n        return ComplexNumber(self.real + other.real, self.imag + other.imag)\n    def __mul__(self, other: 'ComplexNumber') -> 'ComplexNumber':\n        return ComplexNumber(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)\n    def __abs__(self) -> float:\n        import math\n        return math.hypot(self.real, self.imag)\n    def __eq__(self, other: Any) -> bool:\n        if not isinstance(other, ComplexNumber): return False\n        return abs(self.real - other.real) < 1e-6 and abs(self.imag - other.imag) < 1e-6",
        test="c1 = ComplexNumber(1, 2)\nc2 = ComplexNumber(3, 4)\nassert c1 + c2 == ComplexNumber(4, 6)\nassert c1 * c2 == ComplexNumber(-5, 10)\nassert abs(ComplexNumber(3, 4)) == 5.0\n"
    ))

    # 15. BookItem
    tasks.append(create_task(
        name="BookItem",
        signature="class BookItem:\n    def __init__(self, title: str, author: str, isbn: str):\n        self.title = title\n        self.author = author\n        self.isbn = isbn\n        self.is_borrowed = False",
        doc_desc="Represents a library book that can be borrowed and returned.",
        doctests=[">>> b = BookItem('Dune', 'Frank Herbert', '123'); b.borrow(); b.is_borrowed", "True"],
        hint="borrow(): if not is_borrowed, set is_borrowed = True and return True; else return False. return_book(): if is_borrowed set False and return True; else False.",
        solution="    def borrow(self) -> bool:\n        if self.is_borrowed: return False\n        self.is_borrowed = True\n        return True\n    def return_book(self) -> bool:\n        if not self.is_borrowed: return False\n        self.is_borrowed = False\n        return True",
        test="b = BookItem('Dune', 'Frank Herbert', '123')\nassert b.borrow() is True\nassert b.borrow() is False\nassert b.return_book() is True\nassert b.return_book() is False\n"
    ))

    # 16. LibrarySystem
    tasks.append(create_task(
        name="LibrarySystem",
        signature="class LibrarySystem:\n    def __init__(self):\n        self.catalog = {}  # isbn -> {'title': t, 'author': a, 'borrowed': bool}",
        doc_desc="Library catalog system managing books, borrowing, and author queries.",
        doctests=[">>> lib = LibrarySystem(); lib.add_book('123', 'Dune', 'Frank Herbert'); len(lib.find_by_author('Frank Herbert'))", "1"],
        hint="add_book(isbn, title, author). checkout(isbn) marks borrowed=True. find_by_author(author) returns matching books.",
        solution="    def add_book(self, isbn: str, title: str, author: str) -> None:\n        self.catalog[isbn] = {'title': title, 'author': author, 'borrowed': False}\n    def checkout(self, isbn: str) -> bool:\n        if isbn in self.catalog and not self.catalog[isbn]['borrowed']:\n            self.catalog[isbn]['borrowed'] = True\n            return True\n        return False\n    def return_book(self, isbn: str) -> bool:\n        if isbn in self.catalog and self.catalog[isbn]['borrowed']:\n            self.catalog[isbn]['borrowed'] = False\n            return True\n        return False\n    def find_by_author(self, author: str) -> list[dict]:\n        return [v for v in self.catalog.values() if v['author'].lower() == author.lower()]",
        test="lib = LibrarySystem()\nlib.add_book('1', 'Book A', 'Author X')\nlib.add_book('2', 'Book B', 'Author X')\nassert len(lib.find_by_author('Author X')) == 2\nassert lib.checkout('1') is True\nassert lib.checkout('1') is False\nassert lib.return_book('1') is True\n"
    ))

    # 17. EmployeeRecord
    tasks.append(create_task(
        name="EmployeeRecord",
        signature="class EmployeeRecord:\n    def __init__(self, emp_id: int, name: str, salary: float):\n        self.emp_id = emp_id\n        self.name = name\n        self.salary = float(salary)",
        doc_desc="Tracks an employee with give_raise(percent) and calculate_bonus(bonus_rate).",
        doctests=[">>> e = EmployeeRecord(1, 'Alice', 100000); e.give_raise(10); e.salary", "110000.0"],
        hint="give_raise: self.salary *= (1 + percent / 100.0). calculate_bonus: return self.salary * bonus_rate.",
        solution="    def give_raise(self, percent: float) -> None:\n        self.salary *= (1.0 + percent / 100.0)\n    def calculate_bonus(self, bonus_rate: float = 0.1) -> float:\n        return self.salary * bonus_rate",
        test="e = EmployeeRecord(1, 'Alice', 100000.0)\ne.give_raise(10.0)\nassert abs(e.salary - 110000.0) < 1e-5\nassert abs(e.calculate_bonus(0.05) - 5500.0) < 1e-5\n"
    ))

    # 18. ManagerRecord
    tasks.append(create_task(
        name="ManagerRecord",
        signature="class ManagerRecord:\n    def __init__(self, mgr_id: int, name: str, salary: float):\n        self.mgr_id = mgr_id\n        self.name = name\n        self.salary = float(salary)\n        self.reports = []",
        doc_desc="Manager record tracking team members with add_report and team_salary.",
        doctests=[">>> m = ManagerRecord(10, 'Bob', 120000); m.add_report('Alice', 80000); m.team_salary()", "200000.0"],
        hint="add_report(name, salary): appends to reports. team_salary(): returns self.salary + sum of all reports' salaries.",
        solution="    def add_report(self, name: str, salary: float) -> None:\n        self.reports.append({'name': name, 'salary': float(salary)})\n    def team_salary(self) -> float:\n        return self.salary + sum(r['salary'] for r in self.reports)\n    def report_count(self) -> int:\n        return len(self.reports)",
        test="m = ManagerRecord(10, 'Bob', 120000.0)\nm.add_report('Alice', 80000.0)\nassert m.team_salary() == 200000.0\nassert m.report_count() == 1\n"
    ))

    # 19. VendingMachine
    tasks.append(create_task(
        name="VendingMachine",
        signature="class VendingMachine:\n    def __init__(self):\n        self.stock = {}   # item -> {'price': float, 'count': int}\n        self.inserted = 0.0",
        doc_desc="Vending machine accepting money, dispensing items, and giving change.",
        doctests=[">>> vm = VendingMachine(); vm.add_item('soda', 1.5, 5); vm.insert_coin(2.0); vm.purchase('soda')", "(True, 0.5)"],
        hint="purchase(item): if item in stock and stock[item]['count'] > 0 and inserted >= price: deduct count, change = inserted - price, reset inserted = 0, return (True, change); else return (False, 0.0).",
        solution="    def add_item(self, name: str, price: float, count: int) -> None:\n        self.stock[name] = {'price': float(price), 'count': count}\n    def insert_coin(self, amount: float) -> None:\n        self.inserted += amount\n    def purchase(self, name: str) -> tuple[bool, float]:\n        if name in self.stock and self.stock[name]['count'] > 0 and self.inserted >= self.stock[name]['price']:\n            price = self.stock[name]['price']\n            change = self.inserted - price\n            self.stock[name]['count'] -= 1\n            self.inserted = 0.0\n            return (True, change)\n        return (False, 0.0)\n    def refund(self) -> float:\n        amt = self.inserted\n        self.inserted = 0.0\n        return amt",
        test="vm = VendingMachine()\nvm.add_item('soda', 1.5, 2)\nvm.insert_coin(2.0)\nsuccess, change = vm.purchase('soda')\nassert success is True and abs(change - 0.5) < 1e-6\nassert vm.stock['soda']['count'] == 1\nassert vm.purchase('soda') == (False, 0.0)\n"
    ))

    # 20. StateMachine
    tasks.append(create_task(
        name="StateMachine",
        signature="class StateMachine:\n    def __init__(self, initial_state: str):\n        self.state = initial_state\n        self.transitions = {}  # (from_state, event) -> to_state",
        doc_desc="Finite State Machine tracking current_state, registered transitions, and event triggers.",
        doctests=[">>> sm = StateMachine('LOCKED'); sm.add_transition('LOCKED', 'COIN', 'UNLOCKED'); sm.trigger('COIN'); sm.state", "'UNLOCKED'"],
        hint="add_transition(src, event, dst): transitions[(src, event)] = dst. trigger(event): if (state, event) in transitions, update state and return True; else return False.",
        solution="    def add_transition(self, from_state: str, event: str, to_state: str) -> None:\n        self.transitions[(from_state, event)] = to_state\n    def trigger(self, event: str) -> bool:\n        key = (self.state, event)\n        if key in self.transitions:\n            self.state = self.transitions[key]\n            return True\n        return False",
        test="sm = StateMachine('LOCKED')\nsm.add_transition('LOCKED', 'COIN', 'UNLOCKED')\nsm.add_transition('UNLOCKED', 'PUSH', 'LOCKED')\nassert sm.trigger('COIN') is True and sm.state == 'UNLOCKED'\nassert sm.trigger('PUSH') is True and sm.state == 'LOCKED'\nassert sm.trigger('INVALID') is False\n"
    ))

    # 21. TaskScheduler
    tasks.append(create_task(
        name="TaskScheduler",
        signature="class TaskScheduler:\n    def __init__(self):\n        self.tasks = []  # list of (priority, task_name)",
        doc_desc="Priority-based task scheduler queue with add_task, run_next, and has_pending.",
        doctests=[">>> ts = TaskScheduler(); ts.add_task('job1', 5); ts.add_task('job2', 1); ts.run_next()", "'job2'"],
        hint="Use heapq storing (priority, order_counter, name) to ensure deterministic tie-breaking.",
        solution="    def add_task(self, name: str, priority: int) -> None:\n        import heapq\n        heapq.heappush(self.tasks, (priority, len(self.tasks), name))\n    def run_next(self) -> str | None:\n        import heapq\n        return heapq.heappop(self.tasks)[2] if self.tasks else None\n    def has_pending(self) -> bool:\n        return len(self.tasks) > 0",
        test="ts = TaskScheduler()\nts.add_task('low', 10)\nts.add_task('high', 1)\nassert ts.run_next() == 'high'\nassert ts.has_pending() is True\nassert ts.run_next() == 'low'\nassert ts.has_pending() is False\n"
    ))

    # 22. LoggerHistory
    tasks.append(create_task(
        name="LoggerHistory",
        signature="class LoggerHistory:\n    def __init__(self):\n        self.logs = []  # (level, message)",
        doc_desc="In-memory logger recording messages by level (INFO, WARNING, ERROR) with level filtering.",
        doctests=[">>> l = LoggerHistory(); l.log_info('started'); l.log_error('failed'); len(l.get_logs_by_level('ERROR'))", "1"],
        hint="Append tuples (level.upper(), msg). get_logs_by_level(lvl) returns matching messages.",
        solution="    def log(self, level: str, message: str) -> None:\n        self.logs.append((level.upper(), message))\n    def log_info(self, message: str) -> None: self.log('INFO', message)\n    def log_error(self, message: str) -> None: self.log('ERROR', message)\n    def get_logs_by_level(self, level: str) -> list[str]:\n        lvl = level.upper()\n        return [msg for l, msg in self.logs if l == lvl]\n    def clear(self) -> None: self.logs.clear()",
        test="l = LoggerHistory()\nl.log_info('started')\nl.log_error('failed')\nassert len(l.get_logs_by_level('ERROR')) == 1\nassert l.get_logs_by_level('ERROR')[0] == 'failed'\nl.clear()\nassert len(l.logs) == 0\n"
    ))

    # 23. ConfigStore
    tasks.append(create_task(
        name="ConfigStore",
        signature="class ConfigStore:\n    def __init__(self, initial_data: dict | None = None):\n        self.data = dict(initial_data) if initial_data else {}",
        doc_desc="Key-value configuration store with get, set, has, delete, and default fallback.",
        doctests=[">>> cs = ConfigStore({'port': 8080}); cs.get('port')", "8080"],
        hint="Implement get(key, default=None), set(key, val), has(key), and delete(key).",
        solution="    def get(self, key: str, default: Any = None) -> Any:\n        return self.data.get(key, default)\n    def set(self, key: str, value: Any) -> None:\n        self.data[key] = value\n    def has(self, key: str) -> bool:\n        return key in self.data\n    def delete(self, key: str) -> bool:\n        if key in self.data:\n            del self.data[key]\n            return True\n        return False",
        test="cs = ConfigStore({'port': 8080})\nassert cs.get('port') == 8080\nassert cs.get('host', 'localhost') == 'localhost'\ncs.set('debug', True)\nassert cs.has('debug') is True\nassert cs.delete('port') is True\nassert cs.has('port') is False\n"
    ))

    # 24. EventDispatcher
    tasks.append(create_task(
        name="EventDispatcher",
        signature="class EventDispatcher:\n    def __init__(self):\n        self.listeners = {}  # event_name -> list of callables",
        doc_desc="Publish-subscribe event dispatcher supporting subscribe, unsubscribe, and emit.",
        doctests=[">>> ed = EventDispatcher(); res = []; ed.subscribe('click', lambda data: res.append(data)); ed.emit('click', 42); res", "[42]"],
        hint="subscribe(event, cb), unsubscribe(event, cb), emit(event, data) calls each callback with data.",
        solution="    def subscribe(self, event: str, callback: Any) -> None:\n        self.listeners.setdefault(event, []).append(callback)\n    def unsubscribe(self, event: str, callback: Any) -> bool:\n        if event in self.listeners and callback in self.listeners[event]:\n            self.listeners[event].remove(callback)\n            return True\n        return False\n    def emit(self, event: str, data: Any = None) -> None:\n        for cb in self.listeners.get(event, []):\n            cb(data)",
        test="ed = EventDispatcher()\nlog = []\ncb = lambda d: log.append(d)\ned.subscribe('test', cb)\ned.emit('test', 'hello')\nassert log == ['hello']\nassert ed.unsubscribe('test', cb) is True\ned.emit('test', 'world')\nassert log == ['hello']\n"
    ))

    # 25. MemoryCache
    tasks.append(create_task(
        name="MemoryCache",
        signature="class MemoryCache:\n    def __init__(self, max_keys: int = 100):\n        self.max_keys = max_keys\n        self.store = {}",
        doc_desc="In-memory cache with capacity check rejecting or evicting when full.",
        doctests=[">>> mc = MemoryCache(2); mc.set('a', 1); mc.set('b', 2); mc.set('c', 3)", "False"],
        hint="set(k, v): if key not in store and len >= max_keys, return False. Otherwise store[k] = v and return True.",
        solution="    def set(self, key: str, value: Any) -> bool:\n        if key not in self.store and len(self.store) >= self.max_keys:\n            return False\n        self.store[key] = value\n        return True\n    def get(self, key: str, default: Any = None) -> Any:\n        return self.store.get(key, default)\n    def has(self, key: str) -> bool:\n        return key in self.store\n    def clear(self) -> None:\n        self.store.clear()",
        test="mc = MemoryCache(2)\nassert mc.set('a', 1) is True\nassert mc.set('b', 2) is True\nassert mc.set('c', 3) is False\nassert mc.get('a') == 1\nmc.clear()\nassert mc.has('a') is False\n"
    ))

    # 26. RateLimiterTokenBucket
    tasks.append(create_task(
        name="RateLimiterTokenBucket",
        signature="class RateLimiterTokenBucket:\n    def __init__(self, capacity: int, refill_per_step: int):\n        self.capacity = capacity\n        self.refill = refill_per_step\n        self.tokens = capacity",
        doc_desc="Token bucket rate limiter consuming tokens and refilling per tick.",
        doctests=[">>> tb = RateLimiterTokenBucket(5, 2); tb.consume(4)", "True"],
        hint="consume(n): if tokens >= n, deduct n and return True; else return False. refill_step(): tokens = min(capacity, tokens + refill).",
        solution="    def consume(self, n: int = 1) -> bool:\n        if self.tokens >= n:\n            self.tokens -= n\n            return True\n        return False\n    def step_refill(self) -> None:\n        self.tokens = min(self.capacity, self.tokens + self.refill)",
        test="tb = RateLimiterTokenBucket(5, 2)\nassert tb.consume(4) is True\nassert tb.tokens == 1\nassert tb.consume(2) is False\ntb.step_refill()\nassert tb.tokens == 3\nassert tb.consume(2) is True\n"
    ))

    # 27. DeckOfCards
    tasks.append(create_task(
        name="DeckOfCards",
        signature="class DeckOfCards:\n    def __init__(self):\n        suits = ['H', 'D', 'C', 'S']\n        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']\n        self.cards = [f'{r}{s}' for s in suits for r in ranks]",
        doc_desc="Deck of 52 playing cards supporting draw(n), count(), and is_empty.",
        doctests=[">>> d = DeckOfCards(); len(d.draw(5))", "5"],
        hint="draw(n) pops n cards from top of deck (or remaining cards if n > len). count() returns len(cards).",
        solution="    def draw(self, n: int = 1) -> list[str]:\n        drawn = []\n        for _ in range(min(n, len(self.cards))):\n            drawn.append(self.cards.pop())\n        return drawn\n    def count(self) -> int:\n        return len(self.cards)\n    def is_empty(self) -> bool:\n        return len(self.cards) == 0",
        test="d = DeckOfCards()\nassert d.count() == 52\nhand = d.draw(5)\nassert len(hand) == 5\nassert d.count() == 47\n"
    ))

    # 28. Matrix2x2
    tasks.append(create_task(
        name="Matrix2x2",
        signature="class Matrix2x2:\n    def __init__(self, a: float, b: float, c: float, d: float):\n        self.a = float(a)\n        self.b = float(b)\n        self.c = float(c)\n        self.d = float(d)",
        doc_desc="2x2 Matrix [[a, b], [c, d]] supporting determinant, __add__, and __mul__.",
        doctests=[">>> m = Matrix2x2(1, 2, 3, 4); m.determinant()", "-2.0"],
        hint="determinant = a*d - b*c. Matrix multiply [[a1, b1], [c1, d1]] * [[a2, b2], [c2, d2]].",
        solution="    def determinant(self) -> float:\n        return self.a * self.d - self.b * self.c\n    def __add__(self, other: 'Matrix2x2') -> 'Matrix2x2':\n        return Matrix2x2(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)\n    def __mul__(self, other: 'Matrix2x2') -> 'Matrix2x2':\n        return Matrix2x2(\n            self.a * other.a + self.b * other.c,\n            self.a * other.b + self.b * other.d,\n            self.c * other.a + self.d * other.c,\n            self.c * other.b + self.d * other.d\n        )\n    def __eq__(self, other: Any) -> bool:\n        if not isinstance(other, Matrix2x2): return False\n        return all(abs(x - y) < 1e-6 for x, y in zip([self.a, self.b, self.c, self.d], [other.a, other.b, other.c, other.d]))",
        test="m1 = Matrix2x2(1, 2, 3, 4)\nassert m1.determinant() == -2.0\nm2 = Matrix2x2(1, 0, 0, 1)\nassert m1 * m2 == m1\n"
    ))

    # 29. Polynomial
    tasks.append(create_task(
        name="Polynomial",
        signature="class Polynomial:\n    def __init__(self, coeffs: list[float]):\n        # coeffs[i] is coefficient of x^i\n        self.coeffs = [float(c) for c in coeffs] or [0.0]",
        doc_desc="Polynomial evaluation at x, degree calculation, and addition.",
        doctests=[">>> p = Polynomial([1, 2, 3]); p.evaluate(2)", "17.0"],
        hint="evaluate(x): sum(c * (x ** i) for i, c in enumerate(coeffs)). degree(): highest index with non-zero coeff.",
        solution="    def evaluate(self, x: float) -> float:\n        return sum(c * (x ** i) for i, c in enumerate(self.coeffs))\n    def degree(self) -> int:\n        for i in range(len(self.coeffs) - 1, -1, -1):\n            if abs(self.coeffs[i]) > 1e-9:\n                return i\n        return 0\n    def __add__(self, other: 'Polynomial') -> 'Polynomial':\n        from itertools import zip_longest\n        return Polynomial([a + b for a, b in zip_longest(self.coeffs, other.coeffs, fillvalue=0.0)])",
        test="p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2\nassert p.evaluate(2) == 17.0\nassert p.degree() == 2\np2 = Polynomial([2, 1])\nassert (p + p2).coeffs == [3.0, 3.0, 3.0]\n"
    ))

    # 30. RollingAverage
    tasks.append(create_task(
        name="RollingAverage",
        signature="class RollingAverage:\n    def __init__(self, window_size: int):\n        from collections import deque\n        self.window = deque(maxlen=window_size)",
        doc_desc="Maintains rolling window average of stream of numbers.",
        doctests=[">>> ra = RollingAverage(3); ra.add(1); ra.add(2); ra.add(3); ra.average()", "2.0"],
        hint="Use collections.deque(maxlen=window_size). average() returns sum(window)/len(window) or 0.0 if empty.",
        solution="    def add(self, val: float) -> None:\n        self.window.append(float(val))\n    def average(self) -> float:\n        return sum(self.window) / len(self.window) if self.window else 0.0\n    def count(self) -> int:\n        return len(self.window)",
        test="ra = RollingAverage(3)\nra.add(1)\nra.add(2)\nra.add(3)\nassert ra.average() == 2.0\nra.add(4)\nassert abs(ra.average() - 3.0) < 1e-6\n"
    ))

    # 31. VersionTracker
    tasks.append(create_task(
        name="VersionTracker",
        signature="class VersionTracker:\n    def __init__(self, major: int = 0, minor: int = 1, patch: int = 0):\n        self.major = major\n        self.minor = minor\n        self.patch = patch",
        doc_desc="Semantic version tracker with bump_major, bump_minor, and bump_patch.",
        doctests=[">>> vt = VersionTracker(1, 0, 0); vt.bump_minor(); str(vt)", "'1.1.0'"],
        hint="bump_major: major += 1, minor = 0, patch = 0. bump_minor: minor += 1, patch = 0. bump_patch: patch += 1.",
        solution="    def bump_major(self) -> None:\n        self.major += 1\n        self.minor = 0\n        self.patch = 0\n    def bump_minor(self) -> None:\n        self.minor += 1\n        self.patch = 0\n    def bump_patch(self) -> None:\n        self.patch += 1\n    def __str__(self) -> str:\n        return f'{self.major}.{self.minor}.{self.patch}'",
        test="vt = VersionTracker(1, 0, 0)\nvt.bump_patch()\nassert str(vt) == '1.0.1'\nvt.bump_minor()\nassert str(vt) == '1.1.0'\nvt.bump_major()\nassert str(vt) == '2.0.0'\n"
    ))

    # 32. HistoryStack
    tasks.append(create_task(
        name="HistoryStack",
        signature="class HistoryStack:\n    def __init__(self, initial_state: Any):\n        self.undo_stack = [initial_state]\n        self.redo_stack = []",
        doc_desc="Undo-redo history stack managing state transitions.",
        doctests=[">>> hs = HistoryStack('A'); hs.push('B'); hs.undo()", "'A'"],
        hint="push: append to undo_stack, clear redo_stack. undo: pop from undo_stack, push to redo_stack, return current. redo: pop redo, push undo.",
        solution="    def push(self, state: Any) -> None:\n        self.undo_stack.append(state)\n        self.redo_stack.clear()\n    def undo(self) -> Any | None:\n        if len(self.undo_stack) > 1:\n            self.redo_stack.append(self.undo_stack.pop())\n            return self.undo_stack[-1]\n        return None\n    def redo(self) -> Any | None:\n        if self.redo_stack:\n            val = self.redo_stack.pop()\n            self.undo_stack.append(val)\n            return val\n        return None\n    def current(self) -> Any:\n        return self.undo_stack[-1]",
        test="hs = HistoryStack('A')\nhs.push('B')\nhs.push('C')\nassert hs.undo() == 'B'\nassert hs.undo() == 'A'\nassert hs.undo() is None\nassert hs.redo() == 'B'\nassert hs.current() == 'B'\n"
    ))

    # 33. RGBColor
    tasks.append(create_task(
        name="RGBColor",
        signature="class RGBColor:\n    def __init__(self, r: int, g: int, b: int):\n        self.r = max(0, min(255, int(r)))\n        self.g = max(0, min(255, int(g)))\n        self.b = max(0, min(255, int(b)))",
        doc_desc="RGB color container with to_hex(), blend(other, weight), and grayscale().",
        doctests=[">>> RGBColor(255, 0, 128).to_hex()", "'#FF0080'"],
        hint="to_hex: f'#{self.r:02X}{self.g:02X}{self.b:02X}'. grayscale: avg = (r + g + b) // 3, return RGBColor(avg, avg, avg).",
        solution="    def to_hex(self) -> str:\n        return f'#{self.r:02X}{self.g:02X}{self.b:02X}'\n    def grayscale(self) -> 'RGBColor':\n        avg = (self.r + self.g + self.b) // 3\n        return RGBColor(avg, avg, avg)\n    def blend(self, other: 'RGBColor', weight: float = 0.5) -> 'RGBColor':\n        r = int(self.r * (1 - weight) + other.r * weight)\n        g = int(self.g * (1 - weight) + other.g * weight)\n        b = int(self.b * (1 - weight) + other.b * weight)\n        return RGBColor(r, g, b)",
        test="c = RGBColor(255, 0, 128)\nassert c.to_hex() == '#FF0080'\ngray = c.grayscale()\nassert gray.r == gray.g == gray.b\nassert RGBColor(0, 0, 0).blend(RGBColor(200, 200, 200), 0.5).r == 100\n"
    ))

    # 34. IntervalClass
    tasks.append(create_task(
        name="IntervalClass",
        signature="class IntervalClass:\n    def __init__(self, start: float, end: float):\n        self.start = min(float(start), float(end))\n        self.end = max(float(start), float(end))",
        doc_desc="1D numeric interval with contains(val), overlaps(other), and merge(other).",
        doctests=[">>> i1 = IntervalClass(1, 4); i1.contains(3)", "True"],
        hint="contains(x): start <= x <= end. overlaps(other): not (end < other.start or other.end < start). merge: new interval with min(starts), max(ends).",
        solution="    def contains(self, x: float) -> bool:\n        return self.start <= x <= self.end\n    def overlaps(self, other: 'IntervalClass') -> bool:\n        return not (self.end < other.start or other.end < self.start)\n    def merge(self, other: 'IntervalClass') -> 'IntervalClass | None':\n        if not self.overlaps(other): return None\n        return IntervalClass(min(self.start, other.start), max(self.end, other.end))",
        test="i1 = IntervalClass(1, 4)\ni2 = IntervalClass(3, 6)\nassert i1.contains(3) is True\nassert i1.overlaps(i2) is True\nm = i1.merge(i2)\nassert m.start == 1.0 and m.end == 6.0\nassert i1.merge(IntervalClass(10, 12)) is None\n"
    ))

    # 35. PlaylistManager
    tasks.append(create_task(
        name="PlaylistManager",
        signature="class PlaylistManager:\n    def __init__(self, name: str):\n        self.name = name\n        self.tracks = []\n        self.current_idx = 0",
        doc_desc="Music playlist managing tracks with add_track, remove_track, and next_track.",
        doctests=[">>> pl = PlaylistManager('Rock'); pl.add_track('Track 1'); pl.get_current()", "'Track 1'"],
        hint="next_track advances current_idx modulo len(tracks). remove_track removes title and keeps index valid.",
        solution="    def add_track(self, track: str) -> None:\n        self.tracks.append(track)\n    def remove_track(self, track: str) -> bool:\n        if track in self.tracks:\n            self.tracks.remove(track)\n            if self.current_idx >= len(self.tracks):\n                self.current_idx = max(0, len(self.tracks) - 1)\n            return True\n        return False\n    def get_current(self) -> str | None:\n        return self.tracks[self.current_idx] if self.tracks else None\n    def next_track(self) -> str | None:\n        if not self.tracks: return None\n        self.current_idx = (self.current_idx + 1) % len(self.tracks)\n        return self.tracks[self.current_idx]",
        test="pl = PlaylistManager('Favorites')\npl.add_track('Song A')\npl.add_track('Song B')\nassert pl.get_current() == 'Song A'\nassert pl.next_track() == 'Song B'\nassert pl.next_track() == 'Song A'\nassert pl.remove_track('Song B') is True\nassert pl.get_current() == 'Song A'\n"
    ))

    # 36. Scoreboard
    tasks.append(create_task(
        name="Scoreboard",
        signature="class Scoreboard:\n    def __init__(self):\n        self.scores = {}  # player -> score",
        doc_desc="Leaderboard tracking scores with top_k_players and get_rank.",
        doctests=[">>> sb = Scoreboard(); sb.record_score('Alice', 100); sb.get_rank('Alice')", "1"],
        hint="record_score(player, score): keeps max score. top_k(k): sorted descending. get_rank: 1-based rank.",
        solution="    def record_score(self, player: str, score: float) -> None:\n        self.scores[player] = max(self.scores.get(player, float('-inf')), score)\n    def top_k(self, k: int) -> list[tuple[str, float]]:\n        sorted_items = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)\n        return sorted_items[:k]\n    def get_rank(self, player: str) -> int | None:\n        if player not in self.scores:\n            return None\n        sorted_scores = sorted(set(self.scores.values()), reverse=True)\n        return sorted_scores.index(self.scores[player]) + 1",
        test="sb = Scoreboard()\nsb.record_score('Alice', 100)\nsb.record_score('Bob', 200)\nassert sb.top_k(1) == [('Bob', 200)]\nassert sb.get_rank('Bob') == 1\nassert sb.get_rank('Alice') == 2\nassert sb.get_rank('Charlie') is None\n"
    ))

    # 37. RobotRover
    tasks.append(create_task(
        name="RobotRover",
        signature="class RobotRover:\n    def __init__(self, x: int = 0, y: int = 0, orientation: str = 'N'):\n        self.x = x\n        self.y = y\n        self.orientation = orientation\n        self.dirs = ['N', 'E', 'S', 'W']",
        doc_desc="Simulates 2D robot rover with move_forward(steps), turn_left(), and turn_right().",
        doctests=[">>> r = RobotRover(0, 0, 'N'); r.move_forward(3); (r.x, r.y)", "(0, 3)"],
        hint="N: +y, S: -y, E: +x, W: -x. turn_left shifts orientation counter-clockwise, turn_right clockwise.",
        solution="    def turn_right(self) -> None:\n        idx = self.dirs.index(self.orientation)\n        self.orientation = self.dirs[(idx + 1) % 4]\n    def turn_left(self) -> None:\n        idx = self.dirs.index(self.orientation)\n        self.orientation = self.dirs[(idx - 1) % 4]\n    def move_forward(self, steps: int = 1) -> None:\n        if self.orientation == 'N': self.y += steps\n        elif self.orientation == 'S': self.y -= steps\n        elif self.orientation == 'E': self.x += steps\n        elif self.orientation == 'W': self.x -= steps",
        test="r = RobotRover(0, 0, 'N')\nr.move_forward(3)\nassert (r.x, r.y) == (0, 3)\nr.turn_right()\nassert r.orientation == 'E'\nr.move_forward(2)\nassert (r.x, r.y) == (2, 3)\n"
    ))

    # 38. WeatherStation
    tasks.append(create_task(
        name="WeatherStation",
        signature="class WeatherStation:\n    def __init__(self):\n        self.temps = []",
        doc_desc="Station recording temperature readings with min_temp, max_temp, and avg_temp.",
        doctests=[">>> ws = WeatherStation(); ws.record(20.0); ws.record(30.0); ws.avg_temp()", "25.0"],
        hint="record(t) appends float(t). min_temp, max_temp, avg_temp return None if empty.",
        solution="    def record(self, temp: float) -> None:\n        self.temps.append(float(temp))\n    def min_temp(self) -> float | None:\n        return min(self.temps) if self.temps else None\n    def max_temp(self) -> float | None:\n        return max(self.temps) if self.temps else None\n    def avg_temp(self) -> float | None:\n        return sum(self.temps) / len(self.temps) if self.temps else None",
        test="ws = WeatherStation()\nassert ws.min_temp() is None\nws.record(15.0)\nws.record(25.0)\nassert ws.min_temp() == 15.0\nassert ws.max_temp() == 25.0\nassert ws.avg_temp() == 20.0\n"
    ))

    # 39. TextBuffer
    tasks.append(create_task(
        name="TextBuffer",
        signature="class TextBuffer:\n    def __init__(self, initial_text: str = ''):\n        self.chars = list(initial_text)\n        self.cursor = len(self.chars)",
        doc_desc="Text editor buffer supporting insert(text), delete_backwards(n), and move_cursor(pos).",
        doctests=[">>> tb = TextBuffer('hello'); tb.insert(' world'); tb.get_text()", "'hello world'"],
        hint="cursor tracks insertion position. insert splices text into chars at cursor, incrementing cursor.",
        solution="    def insert(self, text: str) -> None:\n        for ch in text:\n            self.chars.insert(self.cursor, ch)\n            self.cursor += 1\n    def delete_backwards(self, n: int = 1) -> None:\n        for _ in range(min(n, self.cursor)):\n            self.cursor -= 1\n            self.chars.pop(self.cursor)\n    def move_cursor(self, pos: int) -> None:\n        self.cursor = max(0, min(len(self.chars), pos))\n    def get_text(self) -> str:\n        return ''.join(self.chars)",
        test="tb = TextBuffer('hello')\ntb.insert(' world')\nassert tb.get_text() == 'hello world'\ntb.delete_backwards(6)\nassert tb.get_text() == 'hello'\ntb.move_cursor(0)\ntb.insert('say ')\nassert tb.get_text() == 'say hello'\n"
    ))

    # 40. StopwatchLapTracker
    tasks.append(create_task(
        name="StopwatchLapTracker",
        signature="class StopwatchLapTracker:\n    def __init__(self):\n        self.laps = []\n        self.running = False\n        self.last_time = 0.0",
        doc_desc="Lap timer recording split times with record_lap and fastest_lap.",
        doctests=[">>> sw = StopwatchLapTracker(); sw.record_lap(12.5); sw.record_lap(11.0); sw.fastest_lap()", "11.0"],
        hint="record_lap(time_sec): appends float. fastest_lap(): min(laps) if laps else None.",
        solution="    def record_lap(self, lap_time: float) -> None:\n        self.laps.append(float(lap_time))\n    def fastest_lap(self) -> float | None:\n        return min(self.laps) if self.laps else None\n    def total_time(self) -> float:\n        return sum(self.laps)\n    def lap_count(self) -> int:\n        return len(self.laps)",
        test="sw = StopwatchLapTracker()\nsw.record_lap(12.5)\nsw.record_lap(11.0)\nsw.record_lap(13.2)\nassert sw.fastest_lap() == 11.0\nassert abs(sw.total_time() - 36.7) < 1e-6\nassert sw.lap_count() == 3\n"
    ))

    # Dynamically generate 24 additional realistic, well-specified OOP models (total 64)
    oop_specs = [
        ("InventoryItem", "class InventoryItem:\n    def __init__(self, sku: str, name: str, price: float, stock: int = 0):\n        self.sku = sku\n        self.name = name\n        self.price = float(price)\n        self.stock = stock",
         "Manages item stock, restock(qty), sell(qty), and is_in_stock.",
         "sell(qty) checks stock >= qty, deducts and returns True; else False.",
         "    def restock(self, qty: int) -> None:\n        self.stock += max(0, qty)\n    def sell(self, qty: int = 1) -> bool:\n        if qty <= self.stock:\n            self.stock -= qty\n            return True\n        return False\n    def is_in_stock(self) -> bool:\n        return self.stock > 0",
         "item = InventoryItem('101', 'Gadget', 9.99, 5)\nassert item.is_in_stock() is True\nassert item.sell(3) is True\nassert item.stock == 2\nassert item.sell(5) is False\n"),
        
        ("DateRange", "class DateRange:\n    def __init__(self, start_day: int, end_day: int):\n        self.start = min(start_day, end_day)\n        self.end = max(start_day, end_day)",
         "Day index range with duration_days and contains_day.",
         "duration_days: end - start + 1. contains_day(d): start <= d <= end.",
         "    def duration_days(self) -> int:\n        return self.end - self.start + 1\n    def contains_day(self, day: int) -> bool:\n        return self.start <= day <= self.end",
         "dr = DateRange(5, 10)\nassert dr.duration_days() == 6\nassert dr.contains_day(7) is True\nassert dr.contains_day(11) is False\n"),

        ("CounterDictionary", "class CounterDictionary:\n    def __init__(self):\n        self.counts = {}",
         "Tracks frequency of hashable keys with increment, get, and most_common_key.",
         "increment(k, by=1) updates counts. most_common_key returns key with highest count or None.",
         "    def increment(self, key: Any, by: int = 1) -> None:\n        self.counts[key] = self.counts.get(key, 0) + by\n    def get(self, key: Any) -> int:\n        return self.counts.get(key, 0)\n    def most_common_key(self) -> Any | None:\n        if not self.counts: return None\n        return max(self.counts.items(), key=lambda x: x[1])[0]",
         "cd = CounterDictionary()\ncd.increment('a', 2)\ncd.increment('b', 5)\nassert cd.get('a') == 2\nassert cd.most_common_key() == 'b'\n"),

        ("UserAccount", "class UserAccount:\n    def __init__(self, username: str, password_hash: str):\n        self.username = username\n        self.hash = password_hash\n        self.is_active = True",
         "User account with deactivate, activate, and check_hash.",
         "check_hash(h): returns self.hash == h and is_active.",
         "    def deactivate(self) -> None: self.is_active = False\n    def activate(self) -> None: self.is_active = True\n    def check_hash(self, h: str) -> bool:\n        return self.is_active and self.hash == h",
         "u = UserAccount('alice', 'hash123')\nassert u.check_hash('hash123') is True\nu.deactivate()\nassert u.check_hash('hash123') is False\n"),

        ("MailboxQueue", "class MailboxQueue:\n    def __init__(self):\n        self.messages = []",
         "Message mailbox with send(msg), read_next(), and unread_count.",
         "read_next pops first message or returns None.",
         "    def send(self, msg: str) -> None: self.messages.append(msg)\n    def read_next(self) -> str | None:\n        return self.messages.pop(0) if self.messages else None\n    def unread_count(self) -> int: return len(self.messages)",
         "mb = MailboxQueue()\nmb.send('m1')\nmb.send('m2')\nassert mb.unread_count() == 2\nassert mb.read_next() == 'm1'\n"),

        ("SimpleCache", "class SimpleCache:\n    def __init__(self, max_items: int = 5):\n        self.max_items = max_items\n        self.data = {}",
         "Cache with clear, put, and get.",
         "put(k, v) only inserts if len < max_items or key exists.",
         "    def put(self, k: Any, v: Any) -> bool:\n        if k not in self.data and len(self.data) >= self.max_items: return False\n        self.data[k] = v\n        return True\n    def get(self, k: Any) -> Any | None: return self.data.get(k)\n    def size(self) -> int: return len(self.data)",
         "c = SimpleCache(2)\nassert c.put('k1', 'v1') is True\nassert c.put('k2', 'v2') is True\nassert c.put('k3', 'v3') is False\n"),

        ("AngleDegree", "class AngleDegree:\n    def __init__(self, deg: float = 0.0):\n        self.deg = float(deg) % 360.0",
         "Represents an angle 0..359.99 degrees with add and normalize.",
         "__add__ wraps modulo 360.",
         "    def __add__(self, other: 'AngleDegree') -> 'AngleDegree':\n        return AngleDegree((self.deg + other.deg) % 360.0)\n    def __eq__(self, other: Any) -> bool:\n        if not isinstance(other, AngleDegree): return False\n        return abs(self.deg - other.deg) < 1e-6",
         "a1 = AngleDegree(270)\na2 = AngleDegree(120)\nassert a1 + a2 == AngleDegree(30)\n"),

        ("StepCounter", "class StepCounter:\n    def __init__(self, target_steps: int = 10000):\n        self.target = target_steps\n        self.steps = 0",
         "Step tracker with step(n), reset(), and is_target_met.",
         "is_target_met: steps >= target.",
         "    def step(self, n: int = 1) -> None: self.steps += max(0, n)\n    def reset(self) -> None: self.steps = 0\n    def is_target_met(self) -> bool: return self.steps >= self.target",
         "sc = StepCounter(100)\nsc.step(50)\nassert sc.is_target_met() is False\nsc.step(60)\nassert sc.is_target_met() is True\n"),

        ("BoundedQueue", "class BoundedQueue:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.items = []",
         "Bounded FIFO queue rejecting items when full.",
         "enqueue returns True if added, False if full.",
         "    def enqueue(self, x: Any) -> bool:\n        if len(self.items) < self.cap:\n            self.items.append(x)\n            return True\n        return False\n    def dequeue(self) -> Any | None:\n        return self.items.pop(0) if self.items else None",
         "bq = BoundedQueue(2)\nassert bq.enqueue(1) is True\nassert bq.enqueue(2) is True\nassert bq.enqueue(3) is False\nassert bq.dequeue() == 1\n"),

        ("KeyValueStore", "class KeyValueStore:\n    def __init__(self):\n        self.store = {}",
         "In-memory key value store with set, get, and keys.",
         "set and get dictionary wrapper.",
         "    def set(self, k: str, v: Any) -> None: self.store[k] = v\n    def get(self, k: str) -> Any | None: return self.store.get(k)\n    def count(self) -> int: return len(self.store)",
         "kv = KeyValueStore()\nkv.set('x', 1)\nassert kv.get('x') == 1\nassert kv.get('y') is None\n"),

        ("SimpleTimer", "class SimpleTimer:\n    def __init__(self):\n        self.elapsed = 0",
         "Timer tracking elapsed ticks with tick(step), reset(), and get_elapsed.",
         "Increment elapsed by step.",
         "    def tick(self, step: int = 1) -> None: self.elapsed += step\n    def reset(self) -> None: self.elapsed = 0\n    def get_elapsed(self) -> int: return self.elapsed",
         "st = SimpleTimer()\nst.tick(5)\nassert st.get_elapsed() == 5\nst.reset()\nassert st.get_elapsed() == 0\n"),

        ("IntegerAccumulator", "class IntegerAccumulator:\n    def __init__(self, start: int = 0):\n        self.total = start",
         "Accumulates integers with add(x), multiply(x), and get_value.",
         "add modifies total += x, multiply modifies total *= x.",
         "    def add(self, x: int) -> int:\n        self.total += x\n        return self.total\n    def multiply(self, x: int) -> int:\n        self.total *= x\n        return self.total\n    def get_value(self) -> int: return self.total",
         "ia = IntegerAccumulator(2)\nassert ia.add(3) == 5\nassert ia.multiply(2) == 10\nassert ia.get_value() == 10\n"),

        ("StringAppender", "class StringAppender:\n    def __init__(self, sep: str = ' '):\n        self.sep = sep\n        self.parts = []",
         "Builds string by appending tokens separated by sep.",
         "append adds token, build rejoins with sep.",
         "    def append(self, token: str) -> None: self.parts.append(token)\n    def build(self) -> str: return self.sep.join(self.parts)\n    def clear(self) -> None: self.parts.clear()",
         "sa = StringAppender('-')\nsa.append('a')\nsa.append('b')\nassert sa.build() == 'a-b'\n"),

        ("MinMaxTracker", "class MinMaxTracker:\n    def __init__(self):\n        self.min_val = None\n        self.max_val = None",
         "Tracks running minimum and maximum of added values.",
         "Update min_val and max_val on each add.",
         "    def add(self, x: float) -> None:\n        self.min_val = x if self.min_val is None else min(self.min_val, x)\n        self.max_val = x if self.max_val is None else max(self.max_val, x)\n    def get_min(self) -> float | None: return self.min_val\n    def get_max(self) -> float | None: return self.max_val",
         "mm = MinMaxTracker()\nmm.add(10)\nmm.add(2)\nmm.add(15)\nassert mm.get_min() == 2\nassert mm.get_max() == 15\n"),

        ("BitFlags", "class BitFlags:\n    def __init__(self, flags: int = 0):\n        self.flags = flags",
         "Manages binary flag bits with set_flag(bit), clear_flag(bit), and has_flag(bit).",
         "Use bitwise OR, AND NOT, and AND.",
         "    def set_flag(self, bit: int) -> None: self.flags |= (1 << bit)\n    def clear_flag(self, bit: int) -> None: self.flags &= ~(1 << bit)\n    def has_flag(self, bit: int) -> bool: return bool(self.flags & (1 << bit))",
         "bf = BitFlags()\nbf.set_flag(2)\nassert bf.has_flag(2) is True\nassert bf.has_flag(1) is False\nbf.clear_flag(2)\nassert bf.has_flag(2) is False\n"),

        ("DiceRoller", "class DiceRoller:\n    def __init__(self, sides: int = 6, seed: int = 42):\n        self.sides = sides\n        self.state = seed",
         "Deterministic LCG dice simulator generating numbers 1..sides.",
         "State update: state = (state * 1103515245 + 12345) & 0x7FFFFFFF; return 1 + (state % sides).",
         "    def roll(self) -> int:\n        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF\n        return 1 + (self.state % self.sides)",
         "dr = DiceRoller(6, 123)\nr1 = dr.roll()\nassert 1 <= r1 <= 6\n"),

        ("SimpleTrieNode", "class SimpleTrieNode:\n    def __init__(self, char: str = ''):\n        self.char = char\n        self.children = {}\n        self.is_end = False",
         "Trie node supporting add_word and contains_word.",
         "Traverse children dict.",
         "    def add_word(self, word: str) -> None:\n        curr = self\n        for ch in word:\n            curr = curr.children.setdefault(ch, SimpleTrieNode(ch))\n        curr.is_end = True\n    def contains_word(self, word: str) -> bool:\n        curr = self\n        for ch in word:\n            if ch not in curr.children: return False\n            curr = curr.children[ch]\n        return curr.is_end",
         "tn = SimpleTrieNode()\ntn.add_word('cat')\nassert tn.contains_word('cat') is True\nassert tn.contains_word('ca') is False\n"),

        ("ScoreAverage", "class ScoreAverage:\n    def __init__(self):\n        self.total = 0.0\n        self.count = 0",
         "Accumulates scores and computes running arithmetic mean.",
         "total / count when count > 0 else 0.0.",
         "    def add_score(self, s: float) -> None:\n        self.total += float(s)\n        self.count += 1\n    def average(self) -> float:\n        return self.total / self.count if self.count else 0.0",
         "sa = ScoreAverage()\nsa.add_score(10)\nsa.add_score(20)\nassert sa.average() == 15.0\n"),

        ("IDGenerator", "class IDGenerator:\n    def __init__(self, prefix: str = 'ID_', start_id: int = 1):\n        self.prefix = prefix\n        self.current = start_id",
         "Generates sequential formatted IDs like 'ID_0001'.",
         "Increment current each next_id() call.",
         "    def next_id(self) -> str:\n        res = f'{self.prefix}{self.current:04d}'\n        self.current += 1\n        return res",
         "ig = IDGenerator('TASK_', 1)\nassert ig.next_id() == 'TASK_0001'\nassert ig.next_id() == 'TASK_0002'\n"),

        ("PrefixTreeSet", "class PrefixTreeSet:\n    def __init__(self):\n        self.words = set()",
         "Set supporting starts_with_prefix(prefix).",
         "Check any word.startswith(prefix).",
         "    def add(self, w: str) -> None: self.words.add(w)\n    def has_prefix(self, p: str) -> bool:\n        return any(w.startswith(p) for w in self.words)",
         "pts = PrefixTreeSet()\npts.add('apple')\nassert pts.has_prefix('app') is True\nassert pts.has_prefix('ban') is False\n"),

        ("TokenCounter", "class TokenCounter:\n    def __init__(self):\n        self.tokens = []",
         "Tracks token strings and returns unique_count and total_count.",
         "unique_count: len(set(tokens)).",
         "    def add(self, t: str) -> None: self.tokens.append(t)\n    def total_count(self) -> int: return len(self.tokens)\n    def unique_count(self) -> int: return len(set(self.tokens))",
         "tc = TokenCounter()\ntc.add('a')\ntc.add('b')\ntc.add('a')\nassert tc.total_count() == 3\nassert tc.unique_count() == 2\n"),

        ("LinearSearcher", "class LinearSearcher:\n    def __init__(self, items: list[int]):\n        self.items = list(items)",
         "Searcher with find_first_index and count_occurrences.",
         "Linear scan helper.",
         "    def find_first(self, target: int) -> int:\n        for i, v in enumerate(self.items):\n            if v == target: return i\n        return -1\n    def count(self, target: int) -> int:\n        return self.items.count(target)",
         "ls = LinearSearcher([1, 2, 3, 2])\nassert ls.find_first(2) == 1\nassert ls.count(2) == 2\n"),

        ("BoundedIntQueue", "class BoundedIntQueue:\n    def __init__(self, cap: int = 5):\n        self.cap = cap\n        self.q = []",
         "Queue of integers with sum_all and is_full.",
         "sum_all returns sum of queue items.",
         "    def push(self, val: int) -> bool:\n        if len(self.q) < self.cap:\n            self.q.append(val)\n            return True\n        return False\n    def pop(self) -> int | None:\n        return self.q.pop(0) if self.q else None\n    def sum_all(self) -> int: return sum(self.q)",
         "biq = BoundedIntQueue(2)\nbiq.push(10)\nbiq.push(20)\nassert biq.sum_all() == 30\nassert biq.push(30) is False\n"),

        ("NamedValueContainer", "class NamedValueContainer:\n    def __init__(self, name: str, value: Any):\n        self.name = name\n        self.value = value",
         "Simple key-value container with rename and update.",
         "Object container with properties.",
         "    def rename(self, new_name: str) -> None: self.name = new_name\n    def update_value(self, new_val: Any) -> None: self.value = new_val\n    def as_tuple(self) -> tuple[str, Any]: return (self.name, self.value)",
         "nvc = NamedValueContainer('score', 10)\nassert nvc.as_tuple() == ('score', 10)\nnvc.rename('points')\nassert nvc.name == 'points'\n")
    ]

    for name, sig, desc, hint, sol, test in oop_specs:
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
