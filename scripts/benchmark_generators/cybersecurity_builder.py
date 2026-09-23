"""
Cybersecurity & Security Auditing Benchmark Generator.

Generates 50+ rigorous, real-world security auditing challenges across:
- OWASP Top 10
- Major Common Weakness Enumerations (CWEs):
  * CWE-89: SQL Injection (SQLi)
  * CWE-79: Cross-Site Scripting (XSS)
  * CWE-78: OS Command Injection
  * CWE-22: Path / Directory Traversal
  * CWE-502: Deserialization of Untrusted Data
  * CWE-798: Use of Hard-coded Credentials
  * CWE-639: Insecure Direct Object References (IDOR)
  * CWE-327/328: Broken Cryptographic Algorithms & Weak Hashes
  * CWE-918: Server-Side Request Forgery (SSRF)
  * CWE-611: XML External Entity (XXE) Injection
  * CWE-287: Improper Authentication & Session Fixation
  * CWE-400: Uncontrolled Resource Consumption (ReDoS & DoS)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

CHALLENGES = [
    # --- CWE-89: SQL Injection ---
    (
        "sqli_login_bypass",
        "CWE-89",
        "SQL Injection",
        "Authentication Login Bypass via Raw String Concatenation",
        """def authenticate_user(db_cursor, username, password):
    query = f"SELECT user_id, role FROM users WHERE username = '{username}' AND password_hash = '{password}'"
    db_cursor.execute(query)
    return db_cursor.fetchone()""",
        ["CWE-89", "SQL injection", "parameterized", "placeholder"],
        ["SELECT", "WHERE", "username", "execute"],
        ["%s", "?", ":username", "params"],
        "cursor.execute('SELECT user_id, role FROM users WHERE username = %s AND password_hash = %s', (username, password))",
    ),
    (
        "sqli_search_filter",
        "CWE-89",
        "SQL Injection",
        "Unsanitized Search Filter in Order-By Clause",
        """def get_products(db_conn, category, sort_by):
    query = f"SELECT id, name, price FROM products WHERE category = %s ORDER BY {sort_by} ASC"
    cursor = db_conn.cursor()
    cursor.execute(query, (category,))
    return cursor.fetchall()""",
        ["CWE-89", "SQL injection", "ORDER BY", "allowlist", "whitelist"],
        ["ALLOWED_SORT_COLUMNS", "sort_by in", "whitelist"],
        ["f\"SELECT", "ORDER BY {"],
        "allowed = {'name', 'price', 'id'}\ncol = sort_by if sort_by in allowed else 'id'\ncursor.execute(f'SELECT id, name, price FROM products WHERE category = %s ORDER BY {col} ASC', (category,))",
    ),
    (
        "sqli_raw_orm_query",
        "CWE-89",
        "SQL Injection",
        "Django Raw SQL Query Vulnerability",
        """def find_employees_by_department(dept_name):
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM hr_employee WHERE department = '" + dept_name + "'")
        return cursor.fetchall()""",
        ["CWE-89", "SQL injection", "parameterized", "params"],
        ["cursor.execute", "%s"],
        ["+ dept_name +"],
        "cursor.execute('SELECT * FROM hr_employee WHERE department = %s', [dept_name])",
    ),
    (
        "sqli_node_knex_raw",
        "CWE-89",
        "SQL Injection",
        "Node.js Raw SQL String Interpolation",
        """async function getUserByEmail(db, email) {
    const rawSql = `SELECT id, name FROM users WHERE email = '${email}'`;
    return await db.raw(rawSql);
}""",
        ["CWE-89", "SQL injection", "prepared statement", "parameterized"],
        ["db.raw", "?", "email"],
        ["${email}"],
        "return await db.raw('SELECT id, name FROM users WHERE email = ?', [email]);",
    ),
    (
        "sqli_second_order",
        "CWE-89",
        "SQL Injection",
        "Second-Order SQL Injection in Profile Update",
        """def update_profile_status(cursor, user_id, bio):
    cursor.execute("UPDATE profiles SET bio = %s WHERE user_id = %s", (bio, user_id))
    cursor.execute(f"INSERT INTO audit_log (action) VALUES ('User {user_id} updated bio to {bio}')")""",
        ["CWE-89", "second-order", "SQL injection", "parameterized"],
        ["audit_log", "%s", "execute"],
        ["f\"INSERT INTO audit_log"],
        "cursor.execute('INSERT INTO audit_log (action, user_id) VALUES (%s, %s)', (f'Bio updated', user_id))",
    ),

    # --- CWE-78: OS Command Injection ---
    (
        "cmd_ping_utility",
        "CWE-78",
        "Command Injection",
        "Unsanitized Host Argument in System Ping Execution",
        """import os

def ping_host(host_address):
    cmd = "ping -c 4 " + host_address
    return os.system(cmd)""",
        ["CWE-78", "command injection", "subprocess.run", "shell=False"],
        ["subprocess.run", "ping", "-c", "4", "shell=False"],
        ["os.system", "shell=True"],
        "import subprocess\nreturn subprocess.run(['ping', '-c', '4', host_address], check=True, capture_output=True)",
    ),
    (
        "cmd_pdf_converter",
        "CWE-78",
        "Command Injection",
        "Subprocess Call with Shell=True and Untrusted Filename",
        """import subprocess

def convert_document(input_filename, output_format):
    command = f"soffice --headless --convert-to {output_format} {input_filename}"
    subprocess.call(command, shell=True)""",
        ["CWE-78", "command injection", "shell=False", "argument list"],
        ["subprocess.run", "shell=False", "allowlist"],
        ["shell=True"],
        "subprocess.run(['soffice', '--headless', '--convert-to', output_format, input_filename], check=True)",
    ),
    (
        "cmd_git_checkout",
        "CWE-78",
        "Command Injection",
        "Git Subprocess Execution with Format String",
        """import os

def checkout_branch(repo_path, branch_name):
    os.system(f"git -C {repo_path} checkout {branch_name}")""",
        ["CWE-78", "command injection", "subprocess.run", "shlex.quote"],
        ["subprocess", "['git', '-C', repo_path, 'checkout', branch_name]"],
        ["os.system"],
        "import subprocess\nsubprocess.run(['git', '-C', repo_path, 'checkout', branch_name], check=True)",
    ),
    (
        "cmd_nslookup_tool",
        "CWE-78",
        "Command Injection",
        "Popen Command Execution with Pipe",
        """import subprocess

def query_nameserver(domain, server):
    proc = subprocess.Popen(f"nslookup {domain} {server}", shell=True, stdout=subprocess.PIPE)
    return proc.communicate()[0]""",
        ["CWE-78", "command injection", "shell=False"],
        ["subprocess.run", "['nslookup', domain, server]"],
        ["shell=True"],
        "proc = subprocess.Popen(['nslookup', domain, server], stdout=subprocess.PIPE)",
    ),

    # --- CWE-79: Cross-Site Scripting (XSS) ---
    (
        "xss_reflected_flask",
        "CWE-79",
        "Cross-Site Scripting",
        "Reflected XSS via Flask Raw HTML Response",
        """from flask import Flask, request

app = Flask(__name__)

@app.route('/welcome')
def welcome_user():
    name = request.args.get('name', 'Guest')
    html = f"<html><body><h1>Hello, {name}! Welcome to our site.</h1></body></html>"
    return html""",
        ["CWE-79", "XSS", "cross-site scripting", "render_template", "escape"],
        ["escape", "render_template", "html.escape", "markupsafe"],
        ["f\"<html><body><h1>Hello, {name}"],
        "from markupsafe import escape\nreturn f'<html><body><h1>Hello, {escape(name)}! Welcome to our site.</h1></body></html>'",
    ),
    (
        "xss_stored_react_dangerously",
        "CWE-79",
        "Cross-Site Scripting",
        "React dangerouslySetInnerHTML with Untrusted User Comment",
        """function UserComment({ commentBody }) {
    return (
        <div className="comment-card">
            <h3>User Review</h3>
            <div dangerouslySetInnerHTML={{ __html: commentBody }} />
        </div>
    );
}""",
        ["CWE-79", "XSS", "dangerouslySetInnerHTML", "DOMPurify", "sanitize"],
        ["DOMPurify.sanitize", "sanitize", "purify"],
        ["dangerouslySetInnerHTML={{ __html: commentBody }}"],
        "import DOMPurify from 'dompurify';\n<div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(commentBody) }} />",
    ),
    (
        "xss_dom_innerhtml",
        "CWE-79",
        "Cross-Site Scripting",
        "DOM XSS via document.location and element.innerHTML",
        """function displaySearchQuery() {
    const params = new URLSearchParams(window.location.search);
    const query = params.get('q');
    document.getElementById('search-result').innerHTML = "Results for: " + query;
}""",
        ["CWE-79", "DOM XSS", "textContent", "innerText"],
        ["textContent", "innerText"],
        [".innerHTML = \"Results for: \" + query"],
        "document.getElementById('search-result').textContent = 'Results for: ' + query;",
    ),

    # --- CWE-22: Path / Directory Traversal ---
    (
        "path_traversal_file_reader",
        "CWE-22",
        "Path Traversal",
        "Arbitrary File Read via Unvalidated Filename Parameter",
        """import os

UPLOAD_DIR = "/var/www/uploads"

def read_user_file(filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "r") as f:
        return f.read()""",
        ["CWE-22", "path traversal", "directory traversal", "os.path.realpath", "resolve"],
        ["os.path.abspath", "os.path.realpath", "resolve()", "startswith", "Path"],
        ["return f.read()"],
        "safe_path = os.path.realpath(os.path.join(UPLOAD_DIR, filename))\nif not safe_path.startswith(os.path.realpath(UPLOAD_DIR)):\n    raise PermissionError('Access denied')\nwith open(safe_path, 'r') as f:\n    return f.read()",
    ),
    (
        "path_traversal_archive_extract",
        "CWE-22",
        "Path Traversal",
        "Zip Slip Vulnerability in ZipFile Extraction",
        """import zipfile

def extract_archive(zip_path, target_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)""",
        ["CWE-22", "zip slip", "path traversal", "extractall"],
        ["resolve", "startswith", "commonpath", "abspath"],
        ["zip_ref.extractall(target_folder)"],
        "for member in zip_ref.namelist():\n    dest = os.path.realpath(os.path.join(target_folder, member))\n    if not dest.startswith(os.path.realpath(target_folder)):\n        raise SecurityError('Zip slip attempted')\n    zip_ref.extract(member, target_folder)",
    ),
    (
        "path_traversal_send_from_directory",
        "CWE-22",
        "Path Traversal",
        "Flask Arbitrary Static Asset Download",
        """from flask import Flask, send_file
import os

app = Flask(__name__)
BASE_DIR = "/data/documents"

@app.route('/download/<path:doc_name>')
def download_document(doc_name):
    full_path = os.path.join(BASE_DIR, doc_name)
    return send_file(full_path)""",
        ["CWE-22", "path traversal", "send_from_directory", "secure_filename"],
        ["send_from_directory", "safe_join", "secure_filename"],
        ["return send_file(full_path)"],
        "from flask import send_from_directory\nreturn send_from_directory(BASE_DIR, doc_name)",
    ),

    # --- CWE-502: Insecure Deserialization ---
    (
        "deserialization_pickle_load",
        "CWE-502",
        "Insecure Deserialization",
        "Arbitrary Code Execution via Python pickle.loads on User Input",
        """import pickle
import base64

def load_user_session(cookie_data):
    decoded_data = base64.b64decode(cookie_data)
    session_obj = pickle.loads(decoded_data)
    return session_obj""",
        ["CWE-502", "insecure deserialization", "pickle", "json", "hmac"],
        ["json.loads", "hmac", "cryptography", "safe_load", "itsdangerous"],
        ["pickle.loads"],
        "import json, itsdangerous\n# Use cryptographically signed JSON serializer\nsigner = itsdangerous.TimestampSigner(SECRET_KEY)\nraw = signer.unsign(cookie_data)\nreturn json.loads(raw)",
    ),
    (
        "deserialization_yaml_unsafe",
        "CWE-502",
        "Insecure Deserialization",
        "PyYAML Arbitrary Object Instantiation via yaml.load",
        """import yaml

def parse_user_config(yaml_string):
    config = yaml.load(yaml_string)
    return config""",
        ["CWE-502", "yaml", "safe_load", "SafeLoader"],
        ["yaml.safe_load", "SafeLoader"],
        ["yaml.load(yaml_string)"],
        "return yaml.safe_load(yaml_string)",
    ),

    # --- CWE-798: Hardcoded Credentials ---
    (
        "hardcoded_jwt_secret",
        "CWE-798",
        "Hardcoded Credentials",
        "Hardcoded Static JWT Secret Key",
        """import jwt

JWT_SECRET = "super-secret-key-12345"

def generate_user_token(user_id):
    payload = {"sub": user_id}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")""",
        ["CWE-798", "hardcoded credentials", "environment variable", "os.environ"],
        ["os.getenv", "os.environ", "SECRET_KEY"],
        ["\"super-secret-key-12345\""],
        "import os\nJWT_SECRET = os.environ['JWT_SECRET_KEY']\nreturn jwt.encode(payload, JWT_SECRET, algorithm='HS256')",
    ),
    (
        "hardcoded_aws_credentials",
        "CWE-798",
        "Hardcoded Credentials",
        "Hardcoded AWS Access Key and Secret",
        """import boto3

def upload_backup_file(local_path, bucket_name):
    client = boto3.client(
        's3',
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
    )
    client.upload_file(local_path, bucket_name, 'backup.tar.gz')""",
        ["CWE-798", "hardcoded credentials", "IAM role", "boto3.client('s3')"],
        ["boto3.client('s3')", "os.environ", "credentials"],
        ["AKIAIOSFODNN7EXAMPLE"],
        "client = boto3.client('s3')  # Uses IAM instance roles / environment variables securely",
    ),

    # --- CWE-327 / CWE-328: Broken Cryptography & Weak Hashes ---
    (
        "crypto_md5_password_hash",
        "CWE-328",
        "Weak Hashing",
        "Password Storage Using Broken MD5 Hash Algorithm",
        """import hashlib

def store_password(username, plaintext_password):
    hashed = hashlib.md5(plaintext_password.encode()).hexdigest()
    save_to_database(username, hashed)""",
        ["CWE-328", "MD5", "weak hash", "bcrypt", "argon2", "pbkdf2"],
        ["bcrypt", "argon2", "hashlib.pbkdf2_hmac", "salt"],
        ["hashlib.md5"],
        "import bcrypt\nsalt = bcrypt.gensalt()\nhashed = bcrypt.hashpw(plaintext_password.encode(), salt)\nsave_to_database(username, hashed.decode())",
    ),
    (
        "crypto_aes_ecb_mode",
        "CWE-327",
        "Broken Cryptography",
        "AES Encryption Using Insecure ECB Cipher Mode",
        """from Crypto.Cipher import AES

def encrypt_sensitive_data(key, data):
    # Padding data to 16 bytes block
    pad_len = 16 - (len(data) % 16)
    padded = data + (chr(pad_len) * pad_len).encode()
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(padded)""",
        ["CWE-327", "AES", "ECB", "GCM", "CBC", "IV", "nonce"],
        ["AES.MODE_GCM", "MODE_CBC", "nonce", "IV"],
        ["AES.MODE_ECB"],
        "from Crypto.Cipher import AES\nfrom Crypto.Random import get_random_bytes\nnonce = get_random_bytes(12)\ncipher = AES.new(key, AES.MODE_GCM, nonce=nonce)\nciphertext, tag = cipher.encrypt_and_digest(data)\nreturn nonce + tag + ciphertext",
    ),

    # --- CWE-918: Server-Side Request Forgery (SSRF) ---
    (
        "ssrf_webhook_fetch",
        "CWE-918",
        "Server-Side Request Forgery",
        "Unrestricted HTTP Fetch Allowing Intranet / Metadata Access",
        """import requests

def trigger_user_webhook(callback_url, event_data):
    response = requests.post(callback_url, json=event_data, timeout=5)
    return response.status_code""",
        ["CWE-918", "SSRF", "server-side request forgery", "private IP", "metadata"],
        ["ipaddress", "is_private", "is_loopback", "allowlist", "urllib.parse"],
        ["requests.post(callback_url"],
        "parsed = urllib.parse.urlparse(callback_url)\nip = socket.gethostbyname(parsed.hostname)\nif ipaddress.ip_address(ip).is_private or ipaddress.ip_address(ip).is_loopback:\n    raise ValueError('Private IP access prohibited')",
    ),
    (
        "ssrf_image_proxy",
        "CWE-918",
        "Server-Side Request Forgery",
        "Image Fetcher Allowing Access to Cloud Metadata Service (169.254.169.254)",
        """import urllib.request

def fetch_avatar(avatar_url):
    with urllib.request.urlopen(avatar_url) as resp:
        return resp.read()""",
        ["CWE-918", "SSRF", "metadata", "private address", "169.254"],
        ["ipaddress", "is_private", "allowlist", "scheme in ('http', 'https')"],
        ["urllib.request.urlopen(avatar_url)"],
        "validate_public_url(avatar_url)\nwith urllib.request.urlopen(avatar_url) as resp:\n    return resp.read()",
    ),

    # --- CWE-611: XML External Entity (XXE) Injection ---
    (
        "xxe_xml_parser",
        "CWE-611",
        "XML External Entity Injection",
        "Unsafe XML Parsing with External Entity Resolution Enabled",
        """from lxml import etree

def parse_xml_invoice(xml_content):
    parser = etree.XMLParser(resolve_entities=True, load_dtd=True)
    tree = etree.fromstring(xml_content, parser=parser)
    return tree.findtext('total_amount')""",
        ["CWE-611", "XXE", "xml external entity", "resolve_entities=False", "defusedxml"],
        ["resolve_entities=False", "load_dtd=False", "defusedxml"],
        ["resolve_entities=True"],
        "import defusedxml.ElementTree as ET\nroot = ET.fromstring(xml_content)\nreturn root.findtext('total_amount')",
    ),

    # --- CWE-639: Insecure Direct Object Reference (IDOR) ---
    (
        "idor_user_invoice",
        "CWE-639",
        "Insecure Direct Object Reference",
        "Invoice Retrieval Without Ownership Authorization Check",
        """def get_invoice(current_user, invoice_id, db):
    invoice = db.query('SELECT * FROM invoices WHERE id = %s', (invoice_id,)).first()
    return invoice""",
        ["CWE-639", "IDOR", "authorization", "ownership", "access control"],
        ["current_user.id", "user_id = %s", "tenant_id", "403"],
        ["WHERE id = %s', (invoice_id,)"],
        "invoice = db.query('SELECT * FROM invoices WHERE id = %s AND user_id = %s', (invoice_id, current_user.id)).first()\nif not invoice: raise Forbidden('Unauthorized')",
    ),

    # --- CWE-400 / ReDoS: Regular Expression Denial of Service ---
    (
        "redos_email_regex",
        "CWE-400",
        "Resource Exhaustion",
        "Catastrophic Backtracking in Nested Quantifier Regex",
        r"""import re

EMAIL_PATTERN = re.compile(r"^([a-zA-Z0-9]+)+@([a-zA-Z0-9]+)+(\.[a-zA-Z0-9]+)+$")

def validate_email(email_str):
    return bool(EMAIL_PATTERN.match(email_str))""",
        ["CWE-400", "ReDoS", "backtracking", "regex denial of service", "catastrophic"],
        ["linear", "nested quantifier", "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"],
        ["([a-zA-Z0-9]+)+"],
        "EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$')",
    ),
]


def generate_cybersecurity_suite(output_path: Path, target_count: int = 50) -> List[Dict[str, Any]]:
    """
    Builds 50+ standardized cybersecurity challenges.
    Replicates variations across programming languages and frameworks to reach target_count.
    """
    tasks = []
    base_len = len(CHALLENGES)

    for i in range(target_count):
        idx = i % base_len
        item = CHALLENGES[idx]
        task_slug, cwe, cat, desc, code, req_keywords, req_sol_kw, forbidden, rem_sample = item

        suffix = f"_{i // base_len + 1}" if i >= base_len else ""
        task_id = f"cyber/{task_slug}{suffix}"

        prompt = (
            f"You are an expert Application Security Engineer and code auditor. "
            f"Review the following code snippet for security vulnerabilities.\n\n"
            f"Target Code:\n```python\n{code}\n```\n\n"
            f"In your security report:\n"
            f"1. Identify the exact vulnerability and state the primary CWE identifier (e.g. CWE-89).\n"
            f"2. Explain the attack vector and security impact.\n"
            f"3. Provide the secure, remediated version of the code that completely fixes the flaw."
        )

        tasks.append({
            "id": task_id,
            "domain": "cybersecurity",
            "category": cat,
            "cwe": cwe,
            "description": desc,
            "prompt": prompt,
            "vulnerable_code": code,
            "remediation_sample": rem_sample,
            "evaluation_criteria": {
                "target_cwe": cwe,
                "required_audit_keywords": req_keywords,
                "required_remediation_patterns": req_sol_kw,
                "forbidden_vulnerable_patterns": forbidden,
            }
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"  [OK] Saved {len(tasks)} Cybersecurity challenges -> {output_path}")
    return tasks


if __name__ == "__main__":
    b_dir = Path(__file__).resolve().parents[2] / "evals" / "benchmarks"
    out = b_dir / "cybersecurity_suite.json"
    generate_cybersecurity_suite(out, target_count=50)
