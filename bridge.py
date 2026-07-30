"""
Bridge between the browser UI and the linear algebra engine (engine.py).
Executed in the same Pyodide global namespace as engine.py.

Exposes two entry points to JS:
  js_compute(op, a_rows_json, k_raw)         -> single-matrix operations
  js_run_notebook(matrices_json, lines_json) -> the expression notebook
"""

import ast
import json
import re


# Rational defines __eq__ but not __hash__, so Python makes it unhashable —
# this breaks any internal set()/dict() use with Rational entries, notably
# Matrix.kernel()/image() deduping basis vectors. Patched here rather than
# in engine.py itself.
def _rational_hash(self):
    return hash((self.numerator, self.denominator))


if Rational.__hash__ is None:
    Rational.__hash__ = _rational_hash


# =========================================================
# Parsing raw cell text into exact engine numbers
# =========================================================

def _parse_number(raw):
    """Integers, fractions ('3/4'), decimals, or simple radicals ('sqrt(2)')."""
    s = str(raw).strip()
    if s == "":
        return Integer(0)

    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return Rational(int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                pass

    try:
        return Integer(int(s))
    except ValueError:
        pass

    try:
        return Rationalize(float(s))
    except ValueError:
        pass

    try:
        namespace = {
            "sqrt": sqrt, "cbrt": cbrt, "nthrt": nthrt,
            "Integer": Integer, "Rational": Rational,
        }
        return eval(s, {"__builtins__": {}}, namespace)
    except Exception:
        raise ValueError("Can't read entry \"%s\"" % raw)


def _build_matrix(rows):
    """rows: list[list[str]] in row-major order (as typed in the UI)."""
    if not rows or not rows[0]:
        raise ValueError("Matrix has no entries")
    parsed = [[_parse_number(cell) for cell in row] for row in rows]
    width = len(parsed[0])
    for row in parsed:
        if len(row) != width:
            raise ValueError("All rows must have the same number of entries")
    columns = list(map(list, zip(*parsed)))
    return Matrix(columns)


def _mat_to_rows(m):
    return [[str(cell) for cell in row] for row in m.rows()]


def _vec_to_list(v):
    return [str(x) for x in v]


# =========================================================
# str() -> LaTeX
# The engine already prints clean unicode math (superscripts,
# "√", Greek letters, "a/b"); we translate that into real LaTeX
# rather than re-deriving formatting from the objects themselves.
# =========================================================

_SUPER_MAP = {ch: str(d) for d, ch in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}
_GREEK_MAP = {
    "λ": r"\lambda", "π": r"\pi", "μ": r"\mu", "α": r"\alpha", "β": r"\beta",
    "θ": r"\theta", "φ": r"\phi", "ω": r"\omega", "Δ": r"\Delta", "Σ": r"\Sigma",
}

_FRAC_RE = re.compile(r'(\([^()]*\)|[A-Za-z0-9_.]+)\s*/\s*(\([^()]*\)|[A-Za-z0-9_.]+)')


def _strip_parens(tok):
    if tok.startswith("(") and tok.endswith(")"):
        return tok[1:-1]
    return tok


def _convert_fractions(text):
    def repl(m):
        num = _strip_parens(m.group(1))
        den = _strip_parens(m.group(2))
        return r"\frac{%s}{%s}" % (num, den)
    prev = None
    while prev != text:
        prev = text
        text = _FRAC_RE.sub(repl, text, count=1)
    return text


def _read_radicand(s, i):
    """s[i] is just past the '√'. Returns (latex, next_index)."""
    n = len(s)
    if i < n and s[i] == "(":
        depth = 0
        j = i
        while j < n:
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0:
                    return to_latex(s[i + 1:j]), j + 1
            j += 1
        return to_latex(s[i:]), n
    j = i
    if j < n and s[j] == "-":
        j += 1
    while j < n and (s[j].isalnum() or s[j] in "./"):
        j += 1
    return to_latex(s[i:j]), j


def to_latex(raw):
    """Convert an engine str() representation into a LaTeX source string."""
    s = str(raw)
    n = len(s)
    out = []
    i = 0
    while i < n:
        ch = s[i]
        if ch in _SUPER_MAP:
            j = i
            digits = ""
            while j < n and s[j] in _SUPER_MAP:
                digits += _SUPER_MAP[s[j]]
                j += 1
            if j < n and s[j] == "√":
                radicand, k = _read_radicand(s, j + 1)
                out.append(r"\sqrt[%s]{%s}" % (digits, radicand))
                i = k
                continue
            out.append("^{%s}" % digits)
            i = j
            continue
        if ch == "√":
            radicand, k = _read_radicand(s, i + 1)
            out.append(r"\sqrt{%s}" % radicand)
            i = k
            continue
        if ch in _GREEK_MAP:
            out.append(_GREEK_MAP[ch])
            i += 1
            continue
        if ch == "−":
            out.append("-")
            i += 1
            continue
        out.append(ch)
        i += 1
    text = _convert_fractions("".join(out))
    return re.sub(r"j\b", "i", text)


def latex_matrix(rows, pivots=None):
    pivot_set = set(tuple(p) for p in (pivots or []))
    body_rows = []
    for r, row in enumerate(rows):
        cells = []
        for c, cell in enumerate(row):
            cell_latex = to_latex(cell)
            if (r, c) in pivot_set:
                cell_latex = r"\boxed{\color{#FF7A50}{%s}}" % cell_latex
            cells.append(cell_latex)
        body_rows.append(" & ".join(cells))
    return r"\begin{bmatrix}" + r" \\ ".join(body_rows) + r"\end{bmatrix}"


def latex_vector(components):
    cells = [to_latex(c) for c in components]
    return r"\begin{bmatrix}" + r" \\ ".join(cells) + r"\end{bmatrix}"


def to_latex_universal(value):
    if isinstance(value, Matrix):
        return latex_matrix(_mat_to_rows(value))
    if isinstance(value, Vector):
        return latex_vector(_vec_to_list(value))
    if isinstance(value, bool):
        return r"\text{true}" if value else r"\text{false}"
    return to_latex(str(value))


# =========================================================
# Single-matrix operations (per matrix-block "quick actions")
# =========================================================

def js_compute(op, a_rows_json, k_raw):
    try:
        A = _build_matrix(json.loads(a_rows_json))
        result = {"ok": True}

        if op == "transpose":
            result["type"] = "matrix"
            result["latex"] = latex_matrix(_mat_to_rows(A.transpose()))
        elif op == "trace":
            result["type"] = "scalar"; result["label"] = "tr"
            result["latex"] = to_latex(A.tr())
        elif op == "det":
            result["type"] = "scalar"; result["label"] = "det"
            result["latex"] = to_latex(A.det())
        elif op == "rank":
            result["type"] = "scalar"; result["label"] = "rank"
            result["latex"] = to_latex(A.rank())
        elif op == "inv":
            result["type"] = "matrix"
            result["latex"] = latex_matrix(_mat_to_rows(A.inv()))
        elif op == "rref":
            R = A.rref()
            result["type"] = "matrix"
            result["latex"] = latex_matrix(_mat_to_rows(R), getattr(R, "_pivots", []))
        elif op == "kernel":
            basis = A.kernel()
            result["type"] = "vectors"; result["label"] = "ker(A) basis"
            result["vectors"] = [latex_vector(_vec_to_list(v)) for v in basis]
        elif op == "image":
            basis = A.image()
            result["type"] = "vectors"; result["label"] = "im(A) basis"
            result["vectors"] = [latex_vector(_vec_to_list(v)) for v in basis]
        elif op == "charpoly":
            p = A.characteristic_polynomial()
            result["type"] = "scalar"; result["label"] = "char. poly"
            result["latex"] = to_latex(p)
        elif op == "power":
            n = int(k_raw)
            result["type"] = "matrix"
            result["latex"] = latex_matrix(_mat_to_rows(A ** n))
        elif op == "properties":
            result["type"] = "properties"
            result["value"] = {
                "square": bool(A.issquare()),
                "symmetric": bool(A.issymetrical()),
                "antisymmetric": bool(A.isantisymertical()),
                "triangular": bool(A.istriangular()),
                "diagonal": bool(A.isdiagonal()),
            }
        else:
            return json.dumps({"ok": False, "error": "Unknown operation '%s'" % op})

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


# =========================================================
# EXPRESSION NOTEBOOK
# =========================================================

_ALLOWED_GLOBALS = [
    "Integer", "Rational", "Radical", "RadicalMonomial", "AlgebraicNumber",
    "sqrt", "cbrt", "nthrt", "Polynomial", "Rationalfraction",
    "Matrix", "Vector", "Constant", "Polynomialvar", "Monomialvar",
    "Rationalfractionvar", "i", "pi", "e", "Greek", "Rationalize",
]

_RESERVED_NAMES = set(_ALLOWED_GLOBALS) | {
    "Identity", "I", "x", "j", "and", "or", "not",
    "det", "rank", "inv", "transpose", "trace", "rref", "kernel", "image", "charpoly",
}


class _IdentityToken:
    """Lets 'Identity' appear bare in an expression: its size is inferred
    from whichever square Matrix it's combined with (see _nb_add/_nb_sub/
    _nb_mul below). Call Identity(n) for an explicit size."""

    def __call__(self, n):
        return Matrix.identity(int(n))

    def __neg__(self):
        raise ValueError("Identity needs a square matrix nearby to size itself (or call Identity(n))")

    def __repr__(self): return "Identity"
    def __str__(self): return "I"


_IDENTITY = _IdentityToken()


def _infer_size(other):
    if isinstance(other, Matrix):
        r, c = other.dim()
        if r != c:
            raise ValueError("Identity is ambiguous next to a non-square matrix")
        return r
    raise ValueError("Identity needs a square matrix nearby to size itself (or call Identity(n))")


def _resolve_identity(a, b):
    if isinstance(a, _IdentityToken):
        a = Matrix.identity(_infer_size(b))
    if isinstance(b, _IdentityToken):
        b = Matrix.identity(_infer_size(a))
    return a, b


def _is_symbolic(v):
    return isinstance(v, (Constant, Monomialvar, Polynomialvar, Rationalfractionvar))


def _poly_has_symbolic_coef(p):
    if isinstance(p, Polynomial):
        return any(_is_symbolic(c) for c in p.lc)
    return _is_symbolic(p)


def _lift_constant(value, other):
    """A bare named Constant (from 'a = 1') can't combine with the
    Polynomial('x') system through its own operators — those two symbolic
    systems in the engine don't interoperate directly. Lifting the Constant
    to a degree-0 polynomial in the same variable routes everything through
    Polynomial's own (working) arithmetic instead."""
    if isinstance(value, Constant) and isinstance(other, (Polynomial, Rationalfraction)):
        cst = other.cst if isinstance(other, Polynomial) else other.numerator.cst
        return Polynomial([value], cst)
    return value


def _prep_pair(a, b):
    a2 = _lift_constant(a, b)
    b2 = _lift_constant(b, a)
    return _resolve_identity(a2, b2)


def _nb_add(a, b):
    a, b = _prep_pair(a, b)
    if isinstance(b, Rationalfraction) and not isinstance(a, Rationalfraction):
        return b + a  # Rationalfraction.__add__ handles Polynomial/scalar; there's no __radd__
    return a + b


def _nb_sub(a, b):
    a, b = _prep_pair(a, b)
    if isinstance(b, Rationalfraction) and not isinstance(a, Rationalfraction):
        return -(b - a)
    return a - b


def _nb_mul(a, b):
    a, b = _prep_pair(a, b)
    if isinstance(b, Rationalfraction) and not isinstance(a, Rationalfraction):
        return b * a
    return a * b


def _nb_div(a, b):
    a, b = _prep_pair(a, b)
    if isinstance(a, Polynomial) and isinstance(b, Polynomial) and \
       (_poly_has_symbolic_coef(a) or _poly_has_symbolic_coef(b)):
        # The engine's gcd-based fraction simplification (gcd_poly) assumes
        # purely numeric coefficients; with a symbolic one it can corrupt a
        # nonzero fraction down to a false "0". Compute both, and only trust
        # the simplified form if it didn't just erase a nonzero numerator.
        raw = Rationalfraction(a, b, _simplify=False)
        try:
            simplified = Rationalfraction(a, b)
        except Exception:
            return raw
        if zero(simplified.numerator) and not zero(raw.numerator):
            return raw
        return simplified
    return a / b


def _nb_pow(base, exponent):
    if isinstance(base, Constant):
        base = Polynomial([base], "x")
    if isinstance(base, _IdentityToken):
        return base
    return base ** exponent


def _fn_det(m): return m.det()
def _fn_rank(m): return m.rank()
def _fn_inv(m): return m.inv()
def _fn_transpose(m): return m.transpose()
def _fn_trace(m): return m.tr()
def _fn_rref(m): return m.rref()
def _fn_kernel(m): return m.kernel()
def _fn_image(m): return m.image()
def _fn_charpoly(m): return m.characteristic_polynomial()


_BINOP_FUNCS = {ast.Add: "_nb_add", ast.Sub: "_nb_sub", ast.Mult: "_nb_mul", ast.Div: "_nb_div"}


class _ExprTransform(ast.NodeTransformer):
    """Two jobs, both needed to keep the engine's own arithmetic happy:
    1) wrap bare int/float literals as Integer/Rationalize so 1/2 stays exact
       (but NOT exponents of ** — Polynomial.__pow__ expects a plain int there).
    2) route +, -, *, /, ** through helper functions — needed so 'Identity'
       can infer its size regardless of order, so a named Constant like 'a'
       can be combined with the Polynomial('x') system, and so symbolic
       fraction division doesn't hit a known engine bug (see _nb_div).
    """

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return node
        if isinstance(node.value, int):
            new = ast.Call(func=ast.Name(id="Integer", ctx=ast.Load()),
                            args=[node], keywords=[])
            return ast.copy_location(new, node)
        if isinstance(node.value, float):
            new = ast.Call(func=ast.Name(id="Rationalize", ctx=ast.Load()),
                            args=[node], keywords=[])
            return ast.copy_location(new, node)
        return node

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            left = self.visit(node.left)
            right = node.right if isinstance(node.right, ast.Constant) else self.visit(node.right)
            call = ast.Call(func=ast.Name(id="_nb_pow", ctx=ast.Load()),
                             args=[left, right], keywords=[])
            return ast.copy_location(call, node)
        self.generic_visit(node)
        fn_name = _BINOP_FUNCS.get(type(node.op))
        if fn_name is None:
            return node
        call = ast.Call(func=ast.Name(id=fn_name, ctx=ast.Load()),
                         args=[node.left, node.right], keywords=[])
        return ast.copy_location(call, node)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _word_break(token, known_names):
    """Try to split e.g. 'PQ' into ['P','Q'] if both are known names."""
    n = len(token)
    dp = [None] * (n + 1)
    dp[0] = []
    for end in range(1, n + 1):
        for start in range(end):
            if dp[start] is not None and token[start:end] in known_names:
                dp[end] = dp[start] + [token[start:end]]
                break
    return dp[n]


def _preprocess_line(line, known_names):
    s = line.strip()
    s = s.replace("^", "**")
    s = re.sub(r"(\d)(?=[A-Za-z_(])", r"\1*", s)     # 2x, 3(A+B)
    s = re.sub(r"(\))(?=[A-Za-z0-9_(])", r"\1*", s)  # (A+B)C, (A)(B)

    def repl(m):
        token = m.group(0)
        if token in known_names or token in _RESERVED_NAMES:
            return token
        parts = _word_break(token, known_names)
        return "*".join(parts) if parts else token

    return _IDENT_RE.sub(repl, s)


_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\s*=(?!=)\s*(.+)$")
_BARE_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _make_namespace(matrices):
    ns = {}
    for name in _ALLOWED_GLOBALS:
        if name in globals():
            ns[name] = globals()[name]
    ns["Identity"] = _IDENTITY
    ns["I"] = _IDENTITY
    ns["x"] = Polynomial([Integer(0), Integer(1)], "x")
    ns["_nb_add"] = _nb_add
    ns["_nb_sub"] = _nb_sub
    ns["_nb_mul"] = _nb_mul
    ns["_nb_div"] = _nb_div
    ns["_nb_pow"] = _nb_pow
    ns["det"] = _fn_det
    ns["rank"] = _fn_rank
    ns["inv"] = _fn_inv
    ns["transpose"] = _fn_transpose
    ns["trace"] = _fn_trace
    ns["rref"] = _fn_rref
    ns["kernel"] = _fn_kernel
    ns["image"] = _fn_image
    ns["charpoly"] = _fn_charpoly
    for name, rows in matrices.items():
        ns[name] = _build_matrix(rows)
    return ns


def js_run_notebook(matrices_json, lines_json):
    matrices = json.loads(matrices_json) if matrices_json else {}
    lines = json.loads(lines_json) if lines_json else []

    try:
        ns = _make_namespace(matrices)
    except Exception as e:
        return json.dumps([{"ok": False, "error": "In matrix input: %s" % e} for _ in lines])

    known_names = set(ns.keys())
    out = []
    unnamed_count = 0

    for raw_line in lines:
        line = (raw_line or "").strip()
        if not line or line.startswith("#"):
            out.append({"ok": True, "empty": True})
            continue

        m = _ASSIGN_RE.match(line)
        assign_name = m.group(1) if m else None
        expr_source = m.group(2) if m else line

        if assign_name and assign_name in matrices:
            out.append({"ok": False, "error": "'%s' is already the name of a matrix" % assign_name})
            continue
        if assign_name and assign_name in _RESERVED_NAMES:
            out.append({"ok": False, "error": "'%s' is reserved and can't be used as a name" % assign_name})
            continue

        if assign_name and _BARE_NUMBER_RE.match(expr_source.strip()):
            # 'a = 1' declares a symbol 'a' that carries the value 1 — it
            # stays symbolic in later expressions (like Desmos sliders),
            # it isn't silently substituted. Write the number itself inline
            # if you want a literal value instead.
            num_text = expr_source.strip()
            numeric = Rationalize(float(num_text)) if "." in num_text else Integer(int(num_text))
            value = Constant(assign_name, numeric)
            ns[assign_name] = value
            known_names.add(assign_name)
            try:
                out.append({"ok": True, "name": assign_name, "latex": to_latex_universal(value)})
            except Exception as e:
                out.append({"ok": False, "error": "Computed, but couldn't render: %s" % e})
            continue

        try:
            expr_source = _preprocess_line(expr_source, known_names)
            tree = ast.parse(expr_source, mode="eval")
            tree = _ExprTransform().visit(tree)
            ast.fix_missing_locations(tree)
            code = compile(tree, "<notebook-line>", "eval")
            safe_builtins = {"abs": abs, "len": len, "range": range, "int": int}
            value = eval(code, {"__builtins__": safe_builtins}, ns)
        except Exception as e:
            out.append({"ok": False, "error": str(e)})
            continue

        if assign_name:
            ns[assign_name] = value
            known_names.add(assign_name)
        else:
            unnamed_count += 1
            ns["ans"] = value
            ns["ans%d" % unnamed_count] = value
            known_names.add("ans"); known_names.add("ans%d" % unnamed_count)

        try:
            latex = to_latex_universal(value)
        except Exception as e:
            out.append({"ok": False, "error": "Computed, but couldn't render: %s" % e})
            continue

        out.append({"ok": True, "name": assign_name, "latex": latex})

    return json.dumps(out)
