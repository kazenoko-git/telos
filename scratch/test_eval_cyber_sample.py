import json
from telos.eval.cyber_eval import evaluate_cybersecurity_challenge

with open('evals/benchmarks/cybersecurity_suite.json') as f:
    bench = json.load(f)

t = [b for b in bench if b['id'] == 'cyber/sqli_raw_orm_query'][0]
completion = """1. **Vulnerability Identification**:  
   The exact vulnerability is **SQL Injection**, with the primary CWE identifier **CWE-89**.

2. **Attack Vector and Security Impact**:  
   The attack vector involves injecting malicious SQL code into the `dept_name` parameter via the web application interface. Since user input is directly concatenated into the SQL query, an attacker can manipulate the input (e.g., `' OR 1=1 --`) to bypass authentication, extract sensitive data, or manipulate the query. The security impact includes unauthorized access, data exfiltration, and potential compromise of employee records.

3. **Secure, Remediated Code**:  
   ```python
   def find_employees_by_department(dept_name):
       from django.db import connection
       with connection.cursor() as cursor:
           cursor.execute("SELECT * FROM hr_employee WHERE department = %s", (dept_name,))
           return cursor.fetchall()
   ```"""

out = evaluate_cybersecurity_challenge(completion, t)
print(json.dumps(out, indent=2))
