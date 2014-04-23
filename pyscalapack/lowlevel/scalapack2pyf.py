

import sys
import re

class ParseException(Exception):
    pass


def parse_file(filename):

    with open(filename, 'r') as f:
        txt = f.read()

    funcname = parse_routine_name(txt)

    secmatch = re.search('\*  Arguments\\n\*  =========\\n\*\\n(.*?)((\*\\n\*  -- )|(\*[^\n]*\\n\*  \=\=\=+)|(\*\\n\\*  Implemented)|(\*\\n\*\s+\w+\:\\n))', txt, flags=re.S)

    if secmatch is None:
        raise ParseException('Could not find argument description.')

    argtxt = secmatch.group(1)

    argmatch = re.compile('^\*  (\w.*)$', flags=re.M)
    args = [ mo.group(1) for mo in argmatch.finditer(argtxt) ]

    return funcname, [ parse_arg(arg) for arg in args ]


def parse_routine_name(string):

    funcname = None

    mo = re.search('void (\w+)_\s*\(', string)
    if mo is not None:
        funcname = mo.group(1)

    mo = re.search('SUBROUTINE (\w+)\s*\(', string)
    if mo is not None:
        funcname = mo.group(1)

    if funcname is None:
        raise ParseException("Could not find routine name.")

    return funcname


def parse_arg(argstring):
    arg_name = re.search('^(\w+)', argstring).group(1)

    # Check if it's an array type
    is_array = re.search('(array|pointer)', argstring) is not None

    # Intent
    is_input = re.search('(input|workspace)', argstring) is not None
    is_output = re.search('output', argstring) is not None

    intent = None

    if is_array:
        intent = 'inout'
    else:
        if is_input:
            if is_output:
                intent = 'inout'
            else:
                intent = 'in'
        else:
            intent = 'out'

    atype_list = ['LOGICAL',
                  'CHARACTER',
                  'CHARACTER*1',
                  'INTEGER',
                  'REAL',
                  'DOUBLE PRECISION',
                  'COMPLEX',
                  'COMPLEX*16']

    atype = None
    for at in atype_list:
        if re.search(re.escape(at), argstring) is not None:
            atype = at

    return {'name': arg_name, 'intent': intent, 'is_array': is_array, 'type': atype}


def fill_missing(args):

    previous = None

    for arg in reversed(args):

        if arg['type'] is None:

            if previous is None:
                print args
                raise ParseException('Could not match argument')

            arg['type'] = previous['type']
            arg['intent'] = previous['intent']
            arg['is_array'] = previous['is_array']

        previous = arg


def args_to_fsig(funcname, args):

    def _arg_list():
        astr = ''

        for arg in args[:-1]:
            astr += (arg['name'] + ', ')
        astr += args[-1]['name']

        return astr

    sig = 'SUBROUTINE %s( %s )\n' % (funcname, _arg_list())

    fill_missing(args)

    for arg in args:
        if not arg['is_array']:
            pat = "  %(type)s, intent(%(intent)s) :: %(name)s\n"
        else:
            pat = "  %(type)s, intent(%(intent)s), dimension(*) :: %(name)s\n"

        sig += (pat % arg)

    sig += 'END SUBROUTINE %s\n' % funcname

    return sig


def scalapack2pyf(inputfile, outputfile=None):

    parsed = parse_file(inputfile)
    sig_pyf = args_to_fsig(*parsed)

    if outputfile is None:
        print sig_pyf
    else:
        with open(outputfile, 'w+') as f:
            f.write(sig_pyf)

if __name__ == '__main__':

    inputfile = sys.argv[1]

    if len(sys.argv) < 3:
        outputfile = None
    else:
        outputfile = sys.argv[2]

    try:
        scalapack2pyf(inputfile, outputfile)
    except ParseException as e:
        print "Error processing %s" % inputfile
        print "  --", e.message
        sys.exit(1)


    
