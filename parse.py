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
    ['INTEGER',         (r'[0-9]+', lambda t: t.copy(value=int(t.value, 0)))],

    ['LPAREN',          r'\('],
    ['RPAREN',          r'\)'],
    ['LBRACKET',        r'\['],
    ['RBRACKET',        r'\]'],
    ['LBRACE',          r'{'],
    ['RBRACE',          r'}'],
    ['COMMA',           r','],

    ['ASSIGN',          r':='],
    ['COLON',           r':'],

    ['PLUS',            r'\+'],
    ['MINUS',           r'-'],
    ['TIMES',           r'\*'],
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

rules = [
    ['identifier', ('IDENTIFIER', lambda p: Identifier(p[0], info=p.get_info(0)))],
    ['integer', ('INTEGER', lambda p: Integer(p[0], info=p.get_info(0)))],
    ['parenthesized', ('LPAREN expr RPAREN', lambda p: p[1])],
    ['atom', 'identifier|integer|parenthesized'],

    ['slice', ('atom LBRACKET expr RBRACKET', lambda p: Slice(p[0], p[2], p[2]))],
    ['slice', ('atom LBRACKET expr COLON expr RBRACKET', lambda p: Slice(p[0], p[2], p[4]))],
    ['not_expr', ('NOT (slice|atom)', lambda p: UnaryOp('NOT', p[1]))],
    ['factor', 'slice|atom|not_expr'],
    ['product', ('factor (TIMES factor)*', reduce_binop)],
    ['sum', ('product ((PLUS|MINUS) product)*', reduce_binop)],
    ['shift_expr', ('sum ((LSHIFT|RSHIFT) sum)*', reduce_binop)],
    ['and_expr', ('shift_expr (AND shift_expr)*', reduce_binop)],
    ['or_expr', ('and_expr (OR and_expr)*', reduce_binop)],
    ['xor_expr', ('or_expr (XOR or_expr)*', reduce_binop)],
    ['comp', ('xor_expr ((EQUALS|LESS_THAN|GREATER_THAN) xor_expr)*', reduce_binop)],
    ['call_args', ('expr (COMMA expr)*', reduce_list)],
    ['call', ('identifier LPAREN call_args RPAREN', lambda p: Call(p[0], p[2]))],
    ['expr', 'call|comp'],

    ['assignment', ('(slice|identifier) ASSIGN expr', lambda p: Assign(p[0], p[2]))],

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
        raise

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

# Parse pseudocode from Intel's XML
def parse_instrinsics(*, path='data.xml', whitelist=None,
        whitelist_regex=None):
    # Parse the XML
    try:
        root = ET.parse(path)
    except FileNotFoundError:
        print('To use this, you must download the intrinsic XML data from:\n'
                'https://software.intel.com/sites/landingpage/IntrinsicsGuide'
                '/files/data-3.4.6.xml', file=sys.stderr)
        sys.exit(1)

    fn_table = {}
    for intrinsic in root.findall('intrinsic'):
        name = intrinsic.attrib['name']
        if whitelist and name not in whitelist:
            continue
        if whitelist_regex and not re.search(whitelist_regex, name):
            continue

        params = [Var(p.attrib['varname'], p.attrib['type'])
                for p in intrinsic.findall('parameter')]
        return_type = intrinsic.attrib['rettype']
        [operation] = [op.text for op in intrinsic.findall('operation')]

        fn_table[name] = parse_operation(name, params, return_type, operation)

    return fn_table
