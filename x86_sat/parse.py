import re
import sys
import xml.etree.ElementTree as ET

from .sprdpl import lex as liblex
from .sprdpl import parse as libparse
from .evaluate import *

################################################################################
## Intel instruction pseudo-code parsing #######################################
################################################################################

# Tokenizer

KEYWORDS = {'IF', 'FI', 'ELSE', 'CASE', 'ESAC', 'OF', 'FOR', 'to', 'TO',
        'downto', 'ENDFOR', 'RETURN', 'DEFINE', 'NOT', 'AND', 'and', 'OR',
        'XOR', 'DO', 'OD', 'WHILE', 'OP'}

def check_ident(t):
    if t.value in KEYWORDS:
        return t.copy(type=t.value.upper())
    return t

tokens = [
    ['IDENTIFIER',      (r'[A-Za-z_][A-Za-z_0-9]*', check_ident)],
    ['INTEGER',         (r'(0x[0-9A-Fa-f]+)|[0-9]+',
        lambda t: t.copy(value=int(t.value, 0)))],

    ['LPAREN',          r'\('],
    ['RPAREN',          r'\)'],
    ['LBRACKET',        r'\['],
    ['RBRACKET',        r'\]'],
    ['LBRACE',          r'{'],
    ['RBRACE',          r'}'],
    ['COMMA',           r','],
    ['PERIOD',          r'\.'],

    ['ASSIGN',          r':='],
    ['COLON',           r':'],
    ['QUESTION',        r'\?'],

    ['PLUS',            r'\+'],
    ['MINUS',           r'-'],
    ['TIMES',           r'\*'],
    ['DIVIDE',          r'/'],
    ['MODULO',          r'%'],
    ['LSHIFT',          r'<<'],
    ['RSHIFT',          r'>>'],
    # XXX bitwise and logical AND/OR, they both get mapped to bitwise for now
    ['AND',             (r'\&\&|\&', lambda t: t.copy(value='AND'))],
    ['OR',              (r'\|\||\|', lambda t: t.copy(value='OR'))],
    ['XOR',             (r'\^', lambda t: t.copy(value='XOR'))],
    ['NOT',             (r'~', lambda t: t.copy(value='NOT'))],

    ['LESS_EQUALS',     r'<='],
    ['LESS_THAN',       r'<'],
    ['GREATER_EQUALS',  r'>='],
    ['GREATER_THAN',    r'>'],
    ['EQUALS',          r'=='],
    ['NOT_EQUALS',      r'!='],

    ['SEMICOLON',       r';'],
    ['WHITESPACE',      (r'([ \t]|\\\n)+', lambda t: None)],
    ['NEWLINE',         r'\n'],
]
lexer = liblex.Lexer(tokens)

# Parser

def reduce_binop(p):
    r = p[0]
    for item in p[1]:
        r = BinaryOp(item[0].upper(), r, item[1])
    return r

def reduce_list(p):
    return p.clone(items=[p[0]] + [item[1] for item in p[1]])

def reduce_trailing(p):
    [r, trailing] = p
    for [fn, *args] in trailing:
        r = fn(r, *args)
    return r

rules = [
    ['identifier', ('IDENTIFIER', lambda p: Identifier(p[0], info=p.get_info(0)))],
    ['integer', ('INTEGER', lambda p: Integer(p[0], info=p.get_info(0)))],
    ['parenthesized', ('LPAREN expr RPAREN', lambda p: p[1])],

    # The "trailing" rules don't return usable data directly, but rather a tuple of
    # a function and extra arguments to be called on the expression
    ['trailing', ('PERIOD IDENTIFIER', lambda p: (Attr, p[1])),
        ('LBRACKET expr RBRACKET', lambda p: (Slice, None, p[1])),
        ('LBRACKET expr COLON expr RBRACKET', lambda p: (Slice, p[1], p[3]))],
    ['compound', ('(identifier|parenthesized) trailing*', reduce_trailing)],

    ['call_args', ('expr (COMMA expr)*', reduce_list)],
    ['call', ('identifier LPAREN call_args RPAREN', lambda p: Call(p[0], p[2]))],
    ['atom', 'integer|call|compound'],

    ['not_expr', ('NOT atom', lambda p: UnaryOp('NOT', p[1]))],
    ['neg_expr', ('MINUS atom', lambda p: UnaryOp('-', p[1]))],
    ['factor', 'atom|not_expr|neg_expr'],
    ['product', ('factor ((TIMES|DIVIDE|MODULO) factor)*', reduce_binop)],
    ['sum', ('product ((PLUS|MINUS) product)*', reduce_binop)],
    ['shift_expr', ('sum ((LSHIFT|RSHIFT) sum)*', reduce_binop)],
    ['and_expr', ('shift_expr (AND shift_expr)*', reduce_binop)],
    ['or_expr', ('and_expr (OR and_expr)*', reduce_binop)],
    ['xor_expr', ('or_expr (XOR or_expr)*', reduce_binop)],
    ['comp', ('xor_expr ((EQUALS|NOT_EQUALS|LESS_THAN|LESS_EQUALS|GREATER_THAN|'
            'GREATER_EQUALS|OP) xor_expr)*', reduce_binop)],
    ['ternary', ('comp QUESTION comp COLON comp',
        lambda p: If(p[0], p[2], p[4]))],
    ['expr', 'ternary|comp'],

    ['assignment', ('compound ASSIGN expr', lambda p: Assign(p[0], p[2]))],
    ['assignment', ('OP ASSIGN expr', lambda p: Assign(Identifier('OP'), p[2]))],

    ['if_trail', ('ELSE if_stmt', lambda p: p[1]),
        ('ELSE stmt_list FI', lambda p: p[1]),
        ('FI', lambda p: Block([]))],
    ['if_stmt', ('IF (expr NEWLINE|parenthesized) stmt_list if_trail',
        lambda p: If(p[1][0] if isinstance(p[1], list) else p[1], p[2], p[3]))],

    ['case_stmt', ('CASE parenthesized OF NEWLINE (integer COLON stmt)+ ESAC',
        lambda p: Case(p[1], [(s[0], s[2]) for s in p[4]]))],

    ['for_stmt', ('FOR identifier ASSIGN expr (TO|DOWNTO) expr NEWLINE stmt_list ENDFOR',
        lambda p: For(p[1], p[3], p[5], p[7], step=(1 if p[4].upper() == 'TO' else -1)))],

    ['while_stmt', ('DO WHILE expr stmt_list OD', lambda p: While(p[2], p[3]))],

    ['return_stmt', ('RETURN expr', lambda p: Return(p[1]))],

    # Some functions are defined with slices on parameters to denote size, e.g.
    # DEFINE fn(a[127:0], b[127:0]). For now, allow trailing attributes/slices but
    # just ignore them
    ['param', ('identifier trailing*', lambda p: p[0])],
    ['params', ('param (COMMA param)*', reduce_list)],
    ['def_stmt', ('DEFINE IDENTIFIER LPAREN params RPAREN LBRACE stmt_list RBRACE',
        lambda p: Function(p[1], p[3], p[6], info=p.get_info(1)))],

    ['stmt', ('[assignment|if_stmt|case_stmt|for_stmt|while_stmt|return_stmt|def_stmt] '
            '(NEWLINE|SEMICOLON)',
        lambda p: p[0])],
    ['stmt_list', ('stmt+', lambda p: Block([s for s in p[0] if s]))],
]

parser = libparse.Parser(rules, 'stmt_list')

def parse_operation(name, params, return_var, return_type, operation):
    operation = operation.replace('\t', ' ' * 4)
    lex_ctx = lexer.input(operation, filename=name)
    try:
        tree = parser.parse(lex_ctx)
    except libparse.ParseError as e:
        e.print()
        sys.exit(1)

    params = [Var(name, type, signed=get_type_signed(etype))
            for [name, type, etype] in params]

    # Create a wrapper function for this intrinsic
    def run(*args, **ctx_args):
        ctx = Context(**ctx_args)
        # Set up arguments
        assert len(args) == len(params), ('wrong number of arguments, got %s, '
                'expected %s') % (len(args), len(params))
        for p, a in zip(params, args):
            a = Value(a, signed=p.signed, width=p.width)
            ctx.set(p.name, a)
        dst = Var(return_var, return_type).eval(ctx)
        ctx.set(return_var, dst)

        # HACK: set MAX based on the result size. This isn't strictly correct,
        # but nearly every intrinsic ends with "dst[MAX:256] := 0" (or similar),
        # which writes all the bits beyond the size of the actual return type.
        # We generally check that the target and value in an assignment have
        # the same number of bits, which is an important sanity check that
        # matters more than this upper-bit zeroing for now. Operations between
        # different vector sizes (something like zmm AND ymm) should be an error
        # anyways, caught by Z3 for differing BitVec sizes.
        ctx.set('MAX', dst.size() - 1)

        # Some intrinsics have a RETURN at top level, so handle that
        try:
            tree.eval(ctx)
        except ReturnExc as e:
            return e.value

        return ctx.get(return_var)

    return Function(name, params, None, return_type=return_type, run=run,
            code=operation, width=get_type_width(return_type))

# Wrapper for metadata parsed from XML data. This supports lazy runtime parsing
# of intrinsics when looked up through getattr (e.g. meta._mm256_set1_epi8(0))
class IntrinsicMetadata:
    def __init__(self, xml_table):
        self.xml_table = xml_table
        self.fn_table = {}

    # Lazily look up intrinsic names, and parse the pseudocode if we haven't yet
    def __getattr__(self, name):
        if name not in self.fn_table:
            args = self.xml_table[name]
            self.fn_table[name] = parse_operation(name, *args)
        return self.fn_table[name]

    # Create a closure class the prepends the given prefix before getattr()
    def prefixed(self, prefix):
        class Prefixed:
            def __getattr__(other, name):
                # Ignore "other"
                return getattr(self, prefix + name)
        return Prefixed()

# Parse the given XML and return an IntrinsicMetadata object
def parse_meta(path):
    try:
        root = ET.parse(path)
    except FileNotFoundError:
        print('To use this, you must download the intrinsic XML data from:\n'
                'https://software.intel.com/sites/landingpage/IntrinsicsGuide'
                '/files/data-latest.xml', file=sys.stderr)
        sys.exit(1)

    version = root.getroot().attrib['version']
    version = tuple(int(x) for x in version.split('.'))

    # Make a metadata dictionary of all intrinsics
    xml_table = {}
    for intrinsic in root.findall('intrinsic'):
        name = intrinsic.attrib['name']
        params = [(p.attrib.get('varname'), p.attrib['type'],
                p.attrib.get('etype', p.attrib['type']))
                for p in intrinsic.findall('parameter')]
        # Return type spec changed in XML as of 3.5.0
        if version < (3, 5, 0):
            [return_var, return_type] = ('dst', intrinsic.attrib['rettype'])
        else:
            [return_var, return_type] = [(r.attrib.get('varname', 'dst'), r.attrib['type'])
                    for r in intrinsic.findall('return')][0]
        operations = [op.text for op in intrinsic.findall('operation')]
        # HACK
        operation = operations[0] if len(operations) == 1 else None

        xml_table[name] = (params, return_var, return_type, operation)

    return IntrinsicMetadata(xml_table)

# Parse the given XML and return a dictionary with intrinsics that match the
# given whitelist (either exact match for 'list' or regex match for 'regex')
def parse_whitelist(metadata, list=None, regex=None):
    if not isinstance(metadata, IntrinsicMetadata):
        assert isinstance(metadata, str)
        metadata = parse_meta(metadata)
    intrinsics = {}
    for name in metadata.xml_table:
        if list and name not in list:
            continue
        if regex and not re.search(regex, name):
            continue
        intrinsics[name] = getattr(metadata, name)
    return intrinsics
