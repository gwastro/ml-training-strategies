import configparser
import numpy as np
import warnings

special_values={
    'pi': np.pi,
    'Pi': np.pi,
    'PI': np.pi,
    'np.pi': np.pi,
    'numpy.pi': np.pi,
    'e': np.e,
    'np.e': np.e,
    'numpy.e': np.e,
    'null': None,
    'none': None,
    'None': None,
    'true': True,
    'True': True,
    'false': False,
    'False': False
    }

known_functions={
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'tan': (np.tan, 1),
    'exp': (np.exp, 1),
    '+': (lambda x, y: x+y, 2),
    '-': (lambda x, y: x-y, 2),
    '*': (lambda x, y: x*y, 2),
    '/': (lambda x, y: x/y, 2)
    }

class BinaryTree(object):
    """A class that implements a basic binary tree that can be sorted by
    a few functions.
    
    Arguments
    ---------
    content : object
        The content this node of a tree should hold.
    parent : {BinaryTree or None, None}
        The BinaryTree-node of which this node is a child.
    
    Attributes
    ----------
    content : object
        The content held by this node.
    parent : BinaryTree or None
        The parent BinaryTree of which this node is a child.
    left : BinaryTree or None
        The left child of this node.
    right : BinaryTree or None
        The right child of this node.
    root : bool
        Whether or not this node is the root of the tree.
    leaf : bool
        True if this node has no children.
    """
    def __init__(self, content, parent=None):
        self.content = content
        self.parent = parent
        self.left = None
        self.right = None
    
    def copy(self):
        """Returns a new instance of itself.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        copy : BinaryTree
            A new instance of the tree from which it was called.
        """
        ret = BinaryTree(self.content, parent=self.parent)
        ret.set_left_child(self.left)
        ret.set_right_child(self.right)
        return ret
    
    def add_left_leaf(self, content):
        """Set the left child of this node to be a BinaryTree without
        any children.
        
        Arguments
        ---------
        content : object
            The content of the child node.
        
        Returns
        -------
        None
        """
        self.left = BinaryTree(content, parent=self)
    
    def add_right_leaf(self, content):
        """Set the right child of this node to be a BinaryTree without
        any children.
        
        Arguments
        ---------
        content : object
            The content of the child node.
        
        Returns
        -------
        None
        """
        self.right = BinaryTree(content, parent=self)
    
    def set_left_child(self, left):
        """Set the left child of this node.
        
        Arguments
        ---------
        left : BinaryTree
            The tree to be the left child of this node.
        
        Returns
        -------
        None
        """
        self.left = left
    
    def set_right_child(self, right):
        """Set the right child of this node.
        
        Arguments
        ---------
        left : BinaryTree
            The tree to be the right child of this node.
        
        Returns
        -------
        None
        """
        self.right = right
    
    @property
    def root(self):
        """Return True if this node has no parent node and thus is a
        root of a tree.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        bool:
            True if this node has no parent node and thus is a root of a
            tree.
        """
        return self.parent is None
    
    @property
    def leaf(self):
        """Return True if the node is a leaf of a tree.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        bool:
            True if the node has no children.
        """
        return self.left is None and self.right is None
    
    def mid_left_right(self):
        """Starting from this node go through the tree and return the
        contents of the node. Go in order node-content -> content of
        left sub-tree -> content of right sub-tree.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        list of object:
            The contents of the tree ordered:
            node content -> left content -> right content
        """
        ret = [self.content]
        if self.left is not None:
            ret.extend(self.left.mid_left_right())
        if self.right is not None:
            ret.extend(self.right.mid_left_right())
        return ret
    
    def left_right_mid(self):     
        """Starting from this node go through the tree and return the
        contents of the node. Go in order content of left sub-tree ->
        content of right sub-tree -> node-content.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        list of object:
            The contents of the tree ordered:
            left content -> right content -> node content
        """
        ret = []
        if self.left is not None:
            ret.extend(self.left.left_right_mid())
        if self.right is not None:
            ret.extend(self.right.left_right_mid())
        ret.append(self.content)
        return ret
    
    def right_left_mid(self):
        """Starting from this node go through the tree and return the
        contents of the node. Go in order content of right sub-tree ->
        content of left sub-tree -> node-content.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        list of object:
            The contents of the tree ordered:
            right content -> left content -> node content
        """
        ret = []
        if self.right is not None:
            ret.extend(self.right.right_left_mid())
        if self.left is not None:
            ret.extend(self.left.right_left_mid())
        ret.append(self.content)
        return ret

class ExpressionString(object):
    """A class that processes strings from config-files.
    
    This class expects the string to contain information that can be
    interpreted as floats, integers or functions applied to those. It
    provides a semi-safe environment that allows to parse only a set of
    pre-defined functions.
    
    This class is meant to handle "easy" cases, i.e. expressions which
    are not too complicated or convoluted. However, the parser is
    written in a way that should be robust to many situations and
    complicated structures. The main limitation is the requirement that
    the output, after applying all functions and replacing all named
    constants, must be convertable to a float or integer.
    
    Arguments
    ---------
    string : {str or None, None}
        The string that should be processed.
    
    Attributes
    ----------
    orig_string : str
        The original input to the class, if it is not overwritten by
        `set_string`.
    string : str
        The string that is worked on by the class. All parsing and
        manipulation to the content of the string is applied to this
        copy of the original string.
    level : nd.array of int
        An internal representation of execution order.
    min : int
        The minimum value of level.
    max : int
        The maximum value of level.
    
    Examples
    --------
    >>> from BnsLib.utils.config import ExpressionString
    >>> exstr = ExpressionString('2')
    >>> exstr.parse()
        2
    >>>
    >>> exstr.set_string('cos(pi)')
    >>> exstr.parse()
        -1.0
    >>>
    >>> exstr.parse_string('sin(2+1)*cos(3+pi)-e')
        -2.578574079359582
    >>>
    >>> exstr.add_named_value('a', 3)
    >>> exstr.parse_string('a')
        3.0
    """
    def __init__(self, string=None):
        self.set_string(string)
        self.known_functions = known_functions
        self.special_values = special_values
    
    def set_string(self, inp):
        """Set the main string of this class. All parsing function will
        apply to this string. (The exception being `parse_string`)
        
        Arguments
        ---------
        inp : str or None
            The string which should be parsed. If set to None a empty
            string will be used.
        
        Returns
        -------
        None
        """
        if inp == None:
            inp = ''
        else:
            inp = str(inp)
        self.orig_string = inp
        self.string = inp.replace(' ', '')
        self.level = np.zeros(len(inp))
    
    def __len__(self):
        """Returns the length of the working-string.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            The length of the working string (in number of characters).
        """
        return len(self.string)
    
    @property
    def min(self):
        """Returns the minimal value of level. The corresponding atomic
        expression will be parsed last and thus is the final action that
        takes place.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            Minimal value of level. 
        """
        return self.level.min()
    
    @property
    def max(self):
        """Returns the maximal value of level. The corresponding atomic
        expression will be parsed fiest and thus is the first action
        that takes place.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            Maximal value of level. 
        """
        return self.level.max()
    
    def add_named_value(self, name, value):
        """Add a value that can be replaced by the parser.
        
        Arguments
        ---------
        name : str
            The name of the value. The parser will search for this
            character sequence in the input string. If found and if the
            string is an atomic expression it will be replaced by the
            value.
        value : float or int
            The value to replace the string by. Its contents are cast to
            be floats.
        
        Returns
        -------
        None
        """
        self.special_values[name] = float(value)
    
    def add_named_values(self, value_dict):
        """Add multiple values to the parser at once.
        
        Arguments
        ---------
        value_dict : dict
            The dictionary that stores the named values. The key, value
            pairs will be passed to add_named_value. (See references
            therein)
        
        Returns
        -------
        None
        """
        for name, value in value_dict.items():
            self.add_named_value(name, value)
    
    def add_function(self, name, function, num_arguments=1):
        """Add a function to the parser.
        
        Arguments
        ---------
        name : str
            The name of the function that is searched for.
        function : callable
            The function to invoke.
        num_arguments : int
            The number of arguments the function takes.
        
        Returns
        -------
        None
        """
        #if num_arguments > 2:
            #msg = 'Currently functions with more than 2 arguments are '
            #msg += 'not supported. They may work, but the behavior of '
            #msg += 'the parser is relatively unpredictable and the '
            #msg += 'success may depend on the order of the arguments.'
            #warnings.warn(msg, RuntimeWarning)
        self.known_functions[name] = (function, num_arguments)
    
    def add_functions(self, function_dict):
        """Add multiple functions at once to the parser.
        
        Arguments
        ---------
        function_dict : dict
            A dictionary containing the keys and function specifier.
            The function specifier must be either a callable or a tuple
            of length two. In the later case the fisr entry must be a
            callable function and the second argument has to be an
            integer specifying the number of arguments the function
            takes. If just a callable is provided the number of
            arguments is expected to be 1.
        
        Returns
        -------
        None
        """
        for name, function_part in function_dict.items():
            try:
                func, num = function_part
            except TypeError:
                func = function_part
                num = 1
            self.add_function(name, func, num_arguments=num)
    
    def parse_commas(self):
        string = []
        ret_level = []
        level = list(self.level)
        for idx, char in enumerate(self.string):
            if char == ',':
                left_bound = 0
                for i in range(idx).__reversed__():
                    if level[i] <= level[idx]:
                        left = i
                level_diff = max(level[left:idx])-level[idx]+1
                for i in range(idx+1, len(self.string)):
                    if level[i] >= level[idx]:
                        level[i] += level_diff
                    else:
                        break
            else:
                string.append(char)
                ret_level.append(level[idx])
        self.string = ''.join(string)
        self.level = np.array(ret_level)
    
    def parse_brackets(self):
        string = []
        level = []
        current_level = 0
        for i in range(len(self.string)):
            if self.string[i] in ['(', '[']:
                current_level += 1
            elif self.string[i] in [')', ']']:
                current_level-= 1
            else:
                level.append(current_level)
                string.append(self.string[i])
        self.string = ''.join(string)
        self.level = np.array(level, dtype=int)
    
    def parse_summation(self):
        insert_zero = []
        for i_sub in range(len(self)):
            i = len(self)-1-i_sub
            if self.string[i] in ['+', '-']:
                left = i
                run = True
                while run:
                    if left <= 0:
                        run = False
                    elif self.level[left] < self.level[i]:
                        left += 1
                        run = False
                    else:
                        left -= 1
                right = i
                run = True
                while run:
                    if right >= len(self):
                        run = False
                    elif self.level[right] < self.level[i]:
                        run = False
                    else:
                        right += 1
                if self.string[i] == '-' and left == i:
                    insert_zero.append([i, self.level[i]+1])
                self.level[left:i] += 1
                self.level[i+1:right] += 1
        
        tmp_string = list(self.string)
        tmp_level = list(self.level)
        while len(insert_zero) > 0:
            curr_idx, curr_level = insert_zero.pop(0)
            for i in range(len(insert_zero)):
                insert_zero[i][0] += 1
            tmp_string.insert(curr_idx, '0')
            curr_level = self.level[curr_idx] + 1
            tmp_level.insert(curr_idx, curr_level)
        self.string = ''.join(tmp_string)
        self.level = np.array(tmp_level, dtype=int)
    
    def parse_product(self):
        for i in range(len(self)):
            if self.string[i] in ['*', '/']:
                left = i
                run = True
                while run:
                    if left <= 0:
                        run = False
                    elif self.level[left] < self.level[i]:
                        left += 1
                        run = False
                    else:
                        left -= 1
                right = i
                run = True
                while run:
                    if right >= len(self):
                        run = False
                    elif self.level[right] < self.level[i]:
                        run = False
                    else:
                        right += 1
                self.level[left:i] += 1
                self.level[i+1:right] += 1
    
    def parse_atomic(self, atomic):
        if atomic in self.known_functions:
            return self.known_functions[atomic]
        elif atomic in self.special_values:
            return self.special_values[atomic]
        else:
            #At this point we assume it is a number
            if '.' in atomic:
                return float(atomic)
            else:
                return int(atomic)
    
    def parse_to_atomic(self):
        atomics = [self.string[0]]
        atomic_levels = [self.level[0]]
        for char, level in zip(self.string[1:], self.level[1:]):
            if atomic_levels[-1] == level:
                atomics[-1] += char
            else:
                atomics.append(char)
                atomic_levels.append(level)
        atomics = [self.parse_atomic(atomic) for atomic in atomics]
        return atomics, atomic_levels
    
    def get_child_indices(self, atomics, atomic_levels, idx):
        left = (None, self.max+1)
        left_part = atomic_levels[:idx]
        if len(left_part) > 0:
            for i in range(1, len(left_part)+1):
                level = atomic_levels[idx - i]
                if level <= atomic_levels[idx]:
                    break
                elif level < left[1]:
                    left = (idx-i, level)
        
        right = (None, self.max+1)
        right_part = atomic_levels[idx+1:]
        if len(right_part) > 0:
            for i in range(1, len(right_part)+1):
                level = atomic_levels[idx+i]
                if level <= atomic_levels[idx]:
                    break
                elif level < right[1]:
                    right = (idx+i, level)
        
        return left[0], right[0]
    
    def tree_from_index(self, atomics, atomic_levels, idx, parent=None):
        if idx is None:
            return None
        ret = BinaryTree(atomics[idx], parent=parent)
        left_idx, right_idx = self.get_child_indices(atomics, atomic_levels, idx)
        ret.set_left_child(self.tree_from_index(atomics, atomic_levels, left_idx, parent=ret))
        ret.set_right_child(self.tree_from_index(atomics, atomic_levels, right_idx, parent=ret))
        return ret
    
    def atomics_to_tree(self, atomics, atomic_levels):
        if len(list(filter(lambda level: level == 0, atomic_levels))) > 1:
            raise RuntimeError('Binary tree needs to have a unique root.')
        
        root = np.where(np.array(atomic_levels, dtype=int) == 0)[0][0]
        tree = self.tree_from_index(atomics, atomic_levels, root)
        
        return tree
    
    def as_tree(self):
        atomics, atomic_levels = self.parse_to_atomic()
        return self.atomics_to_tree(atomics, atomic_levels)
    
    def carry_out_operations(self):
        tree = self.as_tree()
        order_operations = tree.right_left_mid()
        
        stack = []
        for i, operation in enumerate(order_operations):
            if isinstance(operation, tuple):
                args = []
                for i in range(operation[1]):
                    args.append(stack.pop())
                stack.append(operation[0](*args))
            else:
                stack.append(operation)
        
        return stack.pop()
    
    def parse(self):
        if self.orig_string == '':
            return None
        self.parse_brackets()
        self.parse_commas()
        self.parse_summation()
        self.parse_product()
        return self.carry_out_operations()
    
    def parse_string(self, string):
        backup_orig_string = self.orig_string
        backup_string = self.string
        backup_level = self.level
        self.set_string(string)
        res = self.parse()
        self.orig_string = backup_orig_string
        self.string = backup_string
        self.level = backup_level
        return res
    
    def print(self):
        print_string = ''
        for row in range(self.max+1):
            print_string += str(row) + ' '
            for col in range(len(self)):
                if row == self.level[col]:
                    print_string += self.string[col]
                else:
                    print_string += ' '
            print_string += '\n'
        print(print_string)

def get_config_value(inp, constants=None, functions=None):
    """Convert the string returned by ConfigParser to an Python
    expression.
    
    This function uses some special formatting rules. The string may
    contain some special known value like Pi or None. These will be
    converted accordingly. Furthermore, certain operations (+, -, ...)
    and named functions (sin, cos, ...) are supported.
    
    Arguments
    ---------
    inp : str
        The string returned by ConfigParser.get.
    constants : {dict or None, None}
        A dictionary containing constants. The keys specify sub-strings
        of the input string that should be replaced by the values. All
        values are cast to floats.
    functions : {dict or None, None}
        A dictionary containing functions. The keys specify sub-strings
        of the input string that should be replaced by the functions.
        If a function takes more than one argument the value must be a
        tuple, where the first entry is the callable function and the
        second entry specifies the number of arguments the function
        accepts.
    
    Returns
    -------
    expression:
        Tries to interpret the string as a Python expression. This is
        done by parsing the string.
    
    Notes
    -----
    -For details on how the string is parsed see
     BnsLib.utils.config.ExpressionString.
    -To make sure the config-value is understood as a string encapsulate
     it in quotation marks, i.e. "string" or 'string'.
    -The parsing of all strings that are not encapsulated in quotation
     marks is handled by BnsLib.utils.config.ExpressionString. If this
     parser cannot resolve the string to a Python expression the value
     is returned as a string.
    """
    inp = inp.lstrip().rstrip()
    if len(inp) > 1:
        #Test for string
        if inp[0] in ["'", '"'] and inp[-1] in ["'", '"']:
            return inp[1:-1]
        #Test for list
        elif inp[0] == '[' and inp[-1] == ']':
            #TODO: Add this functionality to ExpressionString?
            ret = []
            if len(inp[1:-1]) == 0:
                return ret
            parts = ['']
            in_string = False
            open_brackets = 0
            for char in inp[1:-1]:
                if char == ',' and open_brackets == 0 and not in_string:
                    parts.append('')
                elif char in ['(', '[', '{']:
                    open_brackets += 1
                    parts[-1] += char
                elif char in [')', ']', '{']:
                    open_brackets -= 1
                    parts[-1] += char
                elif char in ['"', "'"]:
                    if in_string:
                        in_string = False
                    else:
                        in_string = True
                    parts[-1] += char
                else:
                    parts[-1] += char
            for pt in parts:
                ret.append(get_config_value(pt,
                                            constants=constants,
                                            functions=functions))
            return ret
        #Test for dict
        elif inp[0] == '{' and inp[-1] == '}':
            #TODO: Add this functionality to ExpressionString?
            ret = {}
            if len(inp[1:-1]) == 0:
                return ret
            parts = ['']
            in_string = False
            open_brackets = 0
            for char in inp[1:-1]:
                if char == ',' and open_brackets == 0 and not in_string:
                    parts.append('')
                elif char in ['(', '[', '{']:
                    open_brackets += 1
                    parts[-1] += char
                elif char in [')', ']', '}']:
                    open_brackets -= 1
                    parts[-1] += char
                elif char in ['"', "'"]:
                    if in_string:
                        in_string = False
                    else:
                        in_string = True
                    parts[-1] += char
                else:
                    parts[-1] += char
            #return parts
            for pt in parts:
                tmp = pt.split(':')
                if len(tmp) != 2:
                    raise ValueError
                key = get_config_value(tmp[0],
                                       constants=constants,
                                       functions=functions)
                value = get_config_value(tmp[1],
                                         constants=constants,
                                         functions=functions)
                
                ret[key] = value
            return ret
    
    if constants is None:
        constants = {}
    
    if functions is None:
        functions = {}
    
    try:
        exp = ExpressionString()
        exp.add_named_values(constants)
        exp.add_functions(functions)
        res = exp.parse_string(inp)
    except:
        res = inp
    return res

def config_to_dict(*file_paths, split_sections=True, constants=None,
                   functions=None):
    """Convert one or multiple config files to a Python dictionary.
    
    Arguments
    ---------
    file_paths : str
        One or multiple paths to config files.
    split_sections : {bool, True}
        If this option is set to True the output dictionary will contain
        a key for every section and the value will be another dictionary
        that contains the key-value-pairs from the config files. If set
        to False a single dictionary will be returned. This dictionary
        contains the key-value-pairs from all sections.
    constants : {dict or None, None}
        A dictionary containing constants. The keys specify sub-strings
        of strings that should be replaced by the values. All values are
        cast to floats.
    functions : {dict or None, None}
        A dictionary containing functions. The keys specify sub-strings
        of strings that should be replaced by the functions. If a
        function takes more than one argument the value must be
        a tuple, where the first entry is the callable function and the
        second entry specifies the number of arguments the function
        accepts.
    
    Returns
    -------
    dict:
        If split_sections is True returns a dictionary where each key
        corresponds to a section.Each entry is furthermore a dictionary
        where each key corresponds to a name in the config file and the
        value is the value of the variable in the config file. The
        values are parsed to be Python data-types and if possible
        converted to int or float.
        If split_sections is False a single dictionary will be returned.
        This dictionary contains the variables and their values from all
        sections.
    
    Notes
    -----
    -See BnsLib.utils.config.get_config_value and
     BnsLib.utils.config.ExpressionString for details on the conversion
     to Python data-types.
    """
    ret = {}
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(*file_paths)
    for sec in conf.sections():
        tmp = {}
        for key in conf[sec].keys():
            tmp[key] = get_config_value(conf[sec][key],
                                        constants=constants,
                                        functions=functions)
        if split_sections:
            ret[sec] = tmp.copy()
        else:
            ret.update(tmp)
    return ret

def dict_to_string_dict(inp_dict):
    return {key: str(val) for (key, val) in inp_dict.items()}
