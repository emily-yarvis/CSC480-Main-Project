import json

filtered = []
keywords = [
    # Control Structures
    "if", "else", "elif", "switch", "case", "default", "while", "for", "do", "break", "continue", "return", "yield",
    # Function & Class Declarations
    "def", "function", "fn", "lambda", "class", "struct", "interface", "constructor", "destructor",
    # Variable Declarations
    "var", "let", "const", "static", "global", "auto", "final", "readonly", "public", "private", "protected",
    # Imports / Modules
    "import", "from", "require", "include", "using", "namespace", "package", "module", "export",
    # Operators and Symbols
    "=", "==", "!=", "===", "!==", "<=", ">=", "++", "--", "->", "=>", "::", "&&", "||", "+=", "-=", "*=", "/=", "%",
    # Data Types
    "int", "float", "double", "char", "string", "bool", "boolean", "list", "dict", "map", "array", "tuple", "enum",
    # Common Built-in Functions
    "print", "input", "len", "append", "pop", "push", "map", "reduce", "filter", "join", "split",
    # Exception Handling
    "try", "catch", "finally", "except", "throw", "raise", "assert",
    # Python-Specific
    "with", "as", "is", "nonlocal", "pass", "del",
    # Java-Specific
    "synchronized", "implements", "extends", "instanceof",
    # C/C++-Specific
    "sizeof", "typedef", "extern", "inline", "volatile",
    # JavaScript-Specific
    "async", "await", "typeof", "this", "new",
    # Rust-Specific
    "match", "impl", "trait", "unwrap", "mut",
    # Go-Specific
    "go", "defer", "select", "chan", "rune",
    # SQL-Specific
    "SELECT", "WHERE", "JOIN", "GROUP BY", "INSERT", "NULL",
    # Symbols
    ";", "//", "#", "{}", "[]", "()", "<>", ":=", "->", "=>", "@", "$"
]


with open("code_alpaca_20k.json", "r") as f:
    convos = json.load(f)
    for convo in convos:
        for keyword in keywords:
            if keyword in convo["output"]:
                filtered.append(convo["output"] + "\n")
                break

with open("code_alpaca_2k.json", "r") as f:
    convos = json.load(f)
    for convo in convos:
        for keyword in keywords:
            if keyword in convo["output"]:
                filtered.append(convo["output"] + "\n")
                break

print(len(filtered))