import json
import tensorflow as tf

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
    "nonlocal", "pass", "del",
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

ai_split = round(len(filtered)*0.8)
train_ai = filtered[:ai_split]
test_ai = filtered[ai_split:]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(<training_code>, <training_labels>,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(<testing_code>, <testing_labels>))

model.summary()