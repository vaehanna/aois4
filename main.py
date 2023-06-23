#4я лабораторная работа, 1е задание -  одноразрядный двоичный сумматор на 3 входа (ОДС-3)
# с представлением выходных функций в СДНФ. .  2е -  n=5, Целиков Фёдор, группа 121702.
import re
import numpy
from enum import Enum

class TYPE_OF_FUNC(Enum):
    CONJUNCTIVE = 0
    DISJUNCTIVE = 1

class ImplicantIsNecessary(Exception):
    pass



class ImplicantIsUnnecessary(Exception):
    pass





def translate_to_implicant(pair, entrance):
    surrounding_translator = [
        [('~a', '~b'), ('~a', '~c'), ('~b', '~c'), ('~b'), ('~c'), ('~a', '~b', '~c')],
        [('~a', 'c'), ('~a', '~b'), ('~b', 'c'), ('c'), ('~b'), ('~a', '~b', 'c')],
        [('~a', 'b'), ('~a', 'c'), ('b', 'c'), ('b'), ('c'), ('~a', 'b', 'c')],
        [('~a', '~c'), ('~a', 'b'), ('b', '~c'), ('~c'), ('b'), ('~a', 'b', '~c')],
        [('a', '~b'), ('a', '~c'), ('~b', '~c'), ('~b'), ('~c'), ('a', '~b', '~c')],
        [('a', 'c'), ('a', '~b'), ('~b', 'c'), ('c'), ('~b'), ('a', '~b', 'c')],
        [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b'), ('c'), ('a', 'b', 'c')],
        [('a', '~c'), ('a', 'b'), ('b', '~c'), ('~c'), ('b'), ('a', 'b', '~c')],
        ]
    index = 0
    for row in range(2):
        for col in range(4):
            if pair == (row, col):
                return surrounding_translator[index][entrance]
            index += 1
    raise Exception("Index is out of range!")



#gets array of implicants as an input
def convert_to_eval(implicants, function_type):
        ##WARN! Could break if you'll change the Enum "TYPE_OF_FUNC"
        #concatenate expression to one string
        if (function_type == TYPE_OF_FUNC.CONJUNCTIVE):
            implicants = [" or ".join(impl) for impl in implicants]
            eval_string = "(" + ") and (".join(implicants) + ")"
        else:
            #convert individual implicants to strings
            implicants = [" and ".join(impl) for impl in implicants]
            eval_string = "(" + ") or (".join(implicants) + ")"
        eval_string = re.sub('~', ' not ', eval_string)
        return eval_string

def convert_to_human(implicants, function_type):
        ##WARN! Could break if you'll change the Enum "TYPE_OF_FUNC"
        #concatenate expression to one string
        if (function_type == TYPE_OF_FUNC.CONJUNCTIVE):
            implicants = ["+".join(impl) for impl in implicants]
            eval_string = "(" +") * (".join(implicants) + ")"
        else:
            #convert individual implicants to strings
            implicants = ["*".join(impl) for impl in implicants]
            eval_string = "(" + ") + (".join(implicants)  + ")"
        eval_string = re.sub(' not ', '~', eval_string)
        return eval_string


def min_rule(inversion):
    buffer = inversion[2: len(inversion) - 1]
    parenthesis_number = 0
    for i in range(len(buffer)):
        if buffer[i] == '(': parenthesis_number += 1
        if buffer[i] == ')': parenthesis_number -= 1
        if buffer[i] == '*' and parenthesis_number == 0:
            buffer = buffer[: i] + '+' + buffer[i + 1:]
        elif buffer[i] == '+' and parenthesis_number == 0:
            buffer = buffer[: i] + '*' + buffer[i + 1:]
    buffer = '~' + buffer
    for i in range(len(buffer)):
        if buffer[i] == '(': parenthesis_number += 1
        if buffer[i] == ')': parenthesis_number -= 1
        if (buffer[i] == '+' or buffer[i] == '*') and parenthesis_number == 0:
            buffer = buffer[:i + 1] + '~' + buffer[i + 1:]
    return buffer

class READING_STATE(Enum):
    READING_SIMPLE = 0
    READING_INVERSION = 1


def check_input(function):
    allowed_symbols = "abc+*~()"
    for i in function:
        if allowed_symbols.find(i) == -1:
            raise Exception('Invalid Input!')


def find_inversion(function):
    check = re.search(r"~\(.+\)", function)
    if check is None: return function
    parenthesis_number = 0
    reading_state = READING_STATE.READING_SIMPLE
    start_index = 0
    end_index = start_index
    inversion = ''
    for i in range(len(function)):
        if function[i] == '~':
            if function[i + 1] != '(':
                inversion += function[i]
                i += 1
                continue
            reading_state = READING_STATE.READING_INVERSION
            start_index = i
            end_index = start_index
            parenthesis_number = 0
            inversion = ''
        if function[i] == '(': parenthesis_number += 1
        if function[i] == ')':
            parenthesis_number -= 1
            if parenthesis_number == 0 and reading_state == READING_STATE.READING_INVERSION:
                reading_state = READING_STATE.READING_SIMPLE
                inversion += function[i]
                end_index = i
                break
        inversion += function[i]
    transformed_inversion = min_rule(inversion)
    if len(inversion) == len(function):
        function = function[: start_index] + transformed_inversion + function[end_index + 1:]
        return function
    else:
        function = function[: start_index] + '(' + transformed_inversion + ')' + function[end_index + 1:]
    return function

def resolve_inversions(function):
    function = find_inversion(function)
    temp = function
    while True:
        temp = find_inversion(function)
        if temp == function:
            break
        else:
            function = temp
    function = normalize(function)
    return function


def normalize(function):
    find_something = re.search(r"(~~)+\w", function)
    while find_something is not None:
        replace_char = find_something.group()
        replace_char = replace_char.replace('~', '')
        function = re.sub(r"(~~)+\w", replace_char, function, 1)
        find_something = re.search(r"(~~)+\w", function)
    return function


def build_truth_table(function):
    function = re.sub(r'\+', ' | ', function)
    function = re.sub(r'\*', ' & ', function)
    table = numpy.zeros(shape=(4, 8))
    solved = function
    i = 0
    for a in range(2):
        a_value = str(a)
        a = re.sub(r'a', a_value, solved)
        for b in range(2):
            b_value = str(b)
            b = re.sub(r'b', b_value, a)
            for c in range(2):
                c_value = str(c)
                c = re.sub(r'c', c_value, b)
                c = re.sub(r'~1', 'False', c)
                c = re.sub(r'~0', 'True', c)
                c = re.sub(r'0', 'False', c)
                c = re.sub(r'1', 'True', c)
                table[0][i] = a_value
                table[1][i] = b_value
                table[2][i] = c_value
                table[3][i] = eval(c)
                i += 1
    return table


def print_truth_table(table):
    options = ['a  ', 'b  ', 'c  ', 'res']
    options_iter = 0
    for i in table:
        row = numpy.array2string(i)
        row = row.replace('[', '')
        row = row.replace('.', '')
        row = row.replace(']', '')
        print(options[options_iter] + ' |' + row)
        print('--------------------')
        options_iter += 1



def make_pdnf(table):
    j = 0
    function = []
    for i in table[3]:
        if i == 1:
            a = b = c = 0
            if table[0][j] == 1:
                a = 'a'
            else:
                a = '~a'
            if table[1][j] == 1:
                b = 'b'
            else:
                b = '~b'
            if table[2][j] == 1:
                c = 'c'
            else:
                c = '~c'
            function.append(a + '*' + b + '*' + c)
        j += 1
    function = " + ".join(function)
    return function



def string_formula(formula):
    option_length = 3
    inside = '*'
    outside = '+'
    substring = []
    for i in range(len(formula)):
        options = []
        for j in range(len(formula[i])):
            if formula[i][j] == 0:
                options.append('!x' + str(j + 1))
            if formula[i][j] == 1:
                options.append('x' + str(j + 1))
        substring.append(inside.join(options))
        if len(substring[-1]) > option_length:
            substring[-1] = '(' + substring[-1] + ')'
    output = outside.join(substring)
    return output


def PDNF(table, optionuments_number):
    formula = []
    optionuments = create_dictionary(optionuments_number)
    for i in range(len(table)):
        if table[i] == 1:
            bracket = []
            for option_index in range(1, optionuments_number + 1):
                bracket.append(optionuments['x' + str(option_index)][i])
            formula.append(bracket)
    return formula


def create_dictionary(optionuments_number):
    dictionary = []
    for i in range(optionuments_number):
        index = i + 1
        same = 2 ** (optionuments_number - index)
        array = [0 for j in range(same)]
        array += [1 for j in range(same)]
        while len(array) < 2 ** (optionuments_number):
            array += array
        dictionary.append(['x' + str(index), array])
    dictionary = dict(dictionary)
    return dictionary


def summator_table():
    optionuments_number = 3
    optionuments = create_dictionary(optionuments_number)
    b = [0 for i in range(len(optionuments['x1']))]
    d = b.copy()
    for i in range(len(optionuments['x1'])):
        sum = optionuments['x1'][i] + optionuments['x2'][i] + optionuments['x3'][i]
        if sum >= 2:
            b[i] = 1
            sum -= 2
        if sum == 1:
            d[i] = 1
    return d, b


def is_mixable(constit1, constit2, option_index, optionuments_number):
    mixability = True
    for i in range(optionuments_number):
        if i != option_index and constit1[i] != constit2[i]:
            mixability = False
            break
        if i == option_index and constit1[i] == constit2[i]:
            mixability = False
            break
    return mixability


def mix(formula, optionuments_number):
    mixd = []
    unmixd = []
    used = [False for i in range(len(formula))]
    for i in range(optionuments_number):
        for j in range(len(formula) - 1):
            for k in range(j + 1, len(formula)):
                if is_mixable(formula[j], formula[k], i, optionuments_number):
                    used[j] = True
                    used[k] = True
                    mixd.append(formula[j].copy())
                    mixd[-1].pop(i)
                    mixd[-1].insert(i, -1)
                    break
    for i in range(len(used)):
        if not (used[i]):
            unmixd.append(formula[i])
    return mixd, unmixd


def replace(values, formula):
    for i in range(len(values)):
        if values[i] == -1:
            missed_value = i
    for i in range(len(formula)):
        if formula[i] != -1 and i != missed_value:
            existing_option = i
    res = []
    res.append(formula[missed_value])
    if formula[existing_option] == values[existing_option]:
        res.append(1)
    else:
        res.append(0)
    return res


def remove(formula):
    new_formula = formula.copy()
    no_change = 1
    i = 0
    while i < len(new_formula):
        res = []
        for other in new_formula:
            if new_formula[i] != other:
                sub = replace(new_formula[i], other)
                if sub[1] == no_change:
                    res.append(sub[0])
        plus, minus = False, False
        for option in res:
            if option == 0: minus = True
            if option == 1: plus = True
        if plus and minus:
            new_formula.pop(i)
        else:
            i += 1
    return new_formula


def delete_identical(formula):
    i = 0
    while i < len(formula) - 1:
        same = False
        for j in range(i + 1, len(formula)):
            if formula[i] == formula[j]:
                same = True
        if same:
            formula.pop(i)
        else:
            i += 1
    return formula


def shorten(formula):
    optionuments_number = len(formula[0])
    i = optionuments_number
    simplified = []
    mixd = formula
    while i > 1:
        mixd, unmixd = mix(mixd, optionuments_number)
        mixd = remove(mixd)
        simplified += unmixd
        i -= 1
    simplified += mixd
    formula = delete_identical(simplified)
    return simplified
def teable():
    capacity = 3
    options = create_dictionary(capacity)
    b = [0 for i in range(len(options['x1']))]
    d = b.copy()
    for i in range(len(options['x1'])):
        sum = options['x1'][i] + options['x2'][i] + options['x3'][i]
        if sum >= 2:
            b[i] = 1
            sum -= 2
        if sum == 1:
            d[i] = 1
    return d, b

def add():
    capacity = 4
    options = create_dictionary(capacity)
    five_bin = [0, 1, 0, 1]
    five_dec = 5
    y = [[0 for m in range(len(options['x1']))] for n in range(capacity)]
    for i in range(len(options['x1']) - five_dec):
        index = capacity
        addone = 0
        while index > 0:
            sum = options['x'+str(index)][i] + five_bin[index-1] + addone
            addone = 0
            if sum >= 2:
                sum -= 2
                addone = 1
            y[index-1][i] = sum
            index -= 1
    return y

print('1st task')
capacity = 3
d, b = teable()
options = create_dictionary(capacity)
print('x1: ' + ' '.join([str(el) for el in options['x1']]))
print('x2: ' + ' '.join([str(el) for el in options['x2']]))
print('x3: ' + ' '.join([str(el) for el in options['x3']]))
print('d:  ' + ' '.join([str(el) for el in d]))
print('b:  ' + ' '.join([str(el) for el in b]))

d_pdnf = PDNF(d, capacity)
print('\nPDNF(d): ' + string_formula(d_pdnf))
d_simplified = shorten(d_pdnf)
d_simplified = string_formula(d_simplified)
print('TDNF(d): ' + d_simplified)
logism_grammar = d_simplified.replace('!', '~')
logism_grammar = logism_grammar.replace('*', '&')
print('TDNF(d) for logism: ' + logism_grammar)

b_pdnf = PDNF(b, capacity)
print('\nPDNF(d): ' + string_formula(b_pdnf))
b_simplified = shorten(b_pdnf)
b_simplified = string_formula(b_simplified)
print('TDNF(d): ' + b_simplified)
logism_grammar = b_simplified.replace('!', '~')
logism_grammar = logism_grammar.replace('*', '&')
print('TDNF(b) for logism: ' + logism_grammar)

print('2nd task, n=5')
capacity = 4
options = create_dictionary(capacity)
for i in range(capacity):
    print('x' + str(i+1) +': ' + ' '.join([str(el) for el in options['x'+str(i+1)]]))
result = add()
print('                 ')
for i in range(capacity):
    print('y' + str(i+1) + ': ' + ' '.join([str(el) for el in result[i]]))

for i in range(capacity):
    y_pdnf = PDNF(result[i], capacity)
    print('\nPDNF(y' + str(i+1) + '): ' + string_formula(y_pdnf))
    y_simplified = shorten(y_pdnf)
    y_simplified = string_formula(y_simplified)
    print('TDNF(y' + str(i+1) + '): ' + y_simplified)
    logism_grammar = y_simplified.replace('!', '~')
    logism_grammar = logism_grammar.replace('*', '&')
    print('TDNF(y' + str(i+1) + ') for logism: ' + logism_grammar)