import importlib.util
import os
import sys

# Load nodes module dynamically since directory has a hyphen
module_path = os.path.join(os.path.dirname(__file__), '..', 'codebase-understanding', 'nodes.py')
spec = importlib.util.spec_from_file_location('nodes', module_path)
nodes = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codebase-understanding'))
spec.loader.exec_module(nodes)


def test_add_line_numbers_basic():
    assert nodes.add_line_numbers('a\nb') == '   1: a\n   2: b'


def test_add_line_numbers_trailing_newline():
    assert nodes.add_line_numbers('a\nb\n') == '   1: a\n   2: b'


def test_add_line_numbers_blank_lines():
    text = 'a\n\nb'
    expected = '   1: a\n   2: \n   3: b'
    assert nodes.add_line_numbers(text) == expected


def test_add_line_numbers_empty():
    assert nodes.add_line_numbers('') == ''
