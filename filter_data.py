import json
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


code = []
labels = []
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
                code.append(convo["output"])
                break

with open("code_alpaca_2k.json", "r") as f:
    convos = json.load(f)
    for convo in convos:
        for keyword in keywords:
            if keyword in convo["output"]:
                code.append(convo["output"])
                break

# print(len(code))
# print(filtered[:10])
labels = [0] * 20357


df = pd.read_parquet('human_code.parquet')
# print(df.head())

code += list(df['code'][:20357])
labels += [1] * 20357

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
X = vectorizer.fit_transform(code) 

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

clf = LinearSVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


ai_sample_code = """def is_palindrome(text):
    \"\"\"
    Returns True if the given string is a palindrome, False otherwise.
    Ignores case and non-alphanumeric characters.
    \"\"\"
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]
"""

	
human_sample_code =  """def to_index(self):
# n.b. creating a new pandas.Index from an old pandas.Index is
# basically free as pandas.Index objects are immutable
assert self.ndim == 1
index = self._data.array
if isinstance(index, pd.MultiIndex):
# set default names for multi-index unnamed levels so that
# we can safely rename dimension / coordinate later
valid_level_names = [name or '{}_level_{}'.format(self.dims[0], i)
for i, name in enumerate(index.names)]
index = index.set_names(valid_level_names)
else:
index = index.set_names(self.name)
return index"""

human_sample_code2 = """
        values = {}

        for index, num in enumerate(nums):
            if target-num in values:
                return [index, values[target-num]]
            values[num] = index
        
        return []
"""

human_sample_code3 = """
        place = 1
        answer = 0
        curr1 = l1
        curr2 = l2

        while curr1 or curr2:
            l1val = curr1.val if curr1 else 0
            l2val = curr2.val if curr2 else 0
            answer += (l1val+l2val) * place
            place *= 10
            curr1 = curr1.next if curr1 else None
            curr2 = curr2.next if curr2 else None
        
        
        answer = list(str(answer))
        ans_list = ListNode(answer[len(answer)-1])
        currans = ans_list
        i = len(answer)-2

        while i >= 0:
            currans.next = ListNode(answer[i])
            currans = currans.next
            i -= 1
        
        return ans_list
"""

my_samples = [
    ai_sample_code,
    human_sample_code,
    human_sample_code2,
    human_sample_code3
]

# Vectorize the new samples using the same vectorizer
X_my_samples = vectorizer.transform(my_samples)

# Predict
y_my_preds = clf.predict(X_my_samples)

# Interpret the output
for code_snippet, label in zip(my_samples, y_my_preds):
    label_str = "Human" if label == 1 else "AI"
    print(f"\n---\nCode:\n{code_snippet}\nPredicted: {label_str}")