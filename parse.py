import re
import sys
import xml.etree.ElementTree as ET

import sprdpl.lex as liblex
import sprdpl.parse as libparse
from evaluate import *

################################################################################
## Intel instruction pseudo-code parsing #######################################
################################################################################

# Tokenizer

KEYWORDS = {'IF', 'FI', 'ELSE', 'CASE', 'ESAC', 'OF', 'FOR', 'to', 'ENDFOR',
        'RETURN', 'DEFINE', 'NOT', 'AND', 'OR', 'XOR'}

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
    ['AND',             r'\&'],
    ['LSHIFT',          r'<<'],
    ['RSHIFT',          r'>>'],

    ['LESS_THAN',       r'<'],
    ['GREATER_THAN',    r'>'],
    ['EQUALS',          r'=='],

    ['NEWLINE',         r'\n'],
    ['WHITESPACE',      (r'[ \t]+', lambda t: None)],
]
lexer = liblex.Lexer(tokens)

# Parser

def reduce_binop(p):
    r = p[0]
    for item in p[1]:
        r = BinaryOp(item[0], r, item[1])
    return r

def reduce_list(p):
    return p.clone(items=[p[0]] + [item[1] for item in p[1]],
        info=[p.info[0]] + [p.info[1][i][1] for i in range(len(p[1]))])

def reduce_trailing(p):
    [r, trailing] = p
    for [fn, *args] in trailing:
        r = fn(r, *args)
    return r

rules = [
    ['identifier', ('IDENTIFIER', lambda p: Identifier(p[0], info=p.get_info(0)))],
    ['integer', ('INTEGER', lambda p: Integer(p[0], info=p.get_info(0)))],
    ['parenthesized', ('LPAREN expr RPAREN', lambda p: p[1])],

    ['trailing', ('PERIOD IDENTIFIER', lambda p: (Attr, p[1])),
        ('LBRACKET expr RBRACKET', lambda p: (Slice, None, p[1])),
        ('LBRACKET expr COLON expr RBRACKET', lambda p: (Slice, p[1], p[3]))],
    ['name', ('identifier trailing*', reduce_trailing)],

    ['call_args', ('expr (COMMA expr)*', reduce_list)],
    ['call', ('identifier LPAREN call_args RPAREN', lambda p: Call(p[0], p[2]))],
    ['atom', 'integer|parenthesized|call|name'],

    ['not_expr', ('NOT atom', lambda p: UnaryOp('NOT', p[1]))],
    ['factor', 'atom|not_expr'],
    ['product', ('factor (TIMES factor)*', reduce_binop)],
    ['sum', ('product ((PLUS|MINUS) product)*', reduce_binop)],
    ['shift_expr', ('sum ((LSHIFT|RSHIFT) sum)*', reduce_binop)],
    ['and_expr', ('shift_expr (AND shift_expr)*', reduce_binop)],
    ['or_expr', ('and_expr (OR and_expr)*', reduce_binop)],
    ['xor_expr', ('or_expr (XOR or_expr)*', reduce_binop)],
    ['comp', ('xor_expr ((EQUALS|LESS_THAN|GREATER_THAN) xor_expr)*', reduce_binop)],
    ['ternary', ('comp QUESTION comp COLON comp',
        lambda p: If(p[0], p[2], p[4]))],
    ['expr', 'ternary|comp'],

    ['assignment', ('name ASSIGN expr', lambda p: Assign(p[0], p[2]))],

    ['if_stmt', ('IF expr NEWLINE stmt_list [ELSE stmt_list] FI',
        lambda p: If(p[1], p[3], p[4][1] if p[4] else Block([])))],

    ['case_stmt', ('CASE parenthesized OF NEWLINE (integer COLON stmt)+ ESAC',
        lambda p: Case(p[1], [(s[0], s[2]) for s in p[4]]))],

    ['for_stmt', ('FOR identifier ASSIGN expr TO expr NEWLINE stmt_list ENDFOR',
        lambda p: For(p[1], p[3], p[5], p[7]))],

    ['return_stmt', ('RETURN expr', lambda p: Return(p[1]))],

    ['params', ('identifier (COMMA identifier)*', reduce_list)],
    ['def_stmt', ('DEFINE IDENTIFIER LPAREN params RPAREN LBRACE stmt_list RBRACE',
        lambda p: Function(p[1], p[3], p[6]))],

    ['stmt', ('[assignment|if_stmt|case_stmt|for_stmt|return_stmt|def_stmt] NEWLINE', lambda p: p[0])],
    ['stmt_list', ('stmt+', lambda p: Block([s for s in p[0] if s]))],
]

parser = libparse.Parser(rules, 'stmt_list')

def parse_operation(name, params, return_type, operation):
    operation = operation.replace('\t', ' ' * 4)
    lex_ctx = lexer.input(operation, filename=name)
    try:
        tree = parser.parse(lex_ctx)
    except libparse.ParseError as e:
        e.print()
        sys.exit(1)

    # Create a wrapper function for this intrinsic
    def run(*args, **ctx_args):
        ctx = Context(**ctx_args)
        # Set up arguments
        assert len(args) == len(params), ('wrong number of arguments, got %s, '
                'expected %s') % (len(args), len(params))
        for p, a in zip(params, args):
            ctx.set(p.name, a)
        dst = Var('dst', return_type).eval(ctx)
        ctx.set('dst', dst)

        # HACK: set MAX based on the result size. This isn't strictly correct,
        # but nearly every intrinsic ends with "dst[MAX:256] := 0" (or similar),
        # which writes all the bits beyond the size of the actual return type.
        # We generally check that the target and value in an assignment have
        # the same number of bits, which is an important sanity check that
        # matters more than this upper-bit zeroing for now. Operations between
        # different vector sizes (something like zmm AND ymm) should be an error
        # anyways, caught by Z3 for differing BitVec sizes.
        ctx.set('MAX', dst.size() - 1)

        tree.eval(ctx)

        return ctx.get('dst')

    return Function(name, params, None, return_type=return_type, run=run,
            code=operation)

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
                '/files/data-3.4.6.xml', file=sys.stderr)
        sys.exit(1)

    # Make a metadata dictionary of all intrinsics
    xml_table = {}
    for intrinsic in root.findall('intrinsic'):
        name = intrinsic.attrib['name']
        params = [Var(p.attrib['varname'], p.attrib['type'])
                for p in intrinsic.findall('parameter')]
        return_type = intrinsic.attrib['rettype']
        operations = [op.text for op in intrinsic.findall('operation')]
        # HACK
        operation = operations[0] if len(operations) == 1 else None

        xml_table[name] = (params, return_type, operation)

    return IntrinsicMetadata(xml_table)

# Parse the given XML and return a dictionary with intrinsics that match the
# given whitelist (either exact match for 'list' or regex match for 'regex')
def parse_whitelist(path, list=None, regex=None):
    metadata = parse_meta(path)
    intrinsics = {}
    for name in metadata.xml_table:
        if list and name not in list:
            continue
        if regex and not re.search(regex, name):
            continue
        intrinsics[name] = getattr(metadata, name)
    return intrinsics
