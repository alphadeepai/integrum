"""Unit tests for MemberFilter and module inspection."""

import pytest
from integrum.filter import MemberFilter
from integrum.inspector import (
    import_target,
    inspect_class,
    inspect_function,
    inspect_module,
)


def sample_adder(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


class Calculator:
    """Simple calculator class."""

    def __init__(self, base: int = 0):
        self.base = base

    def add(self, x: int) -> int:
        """Add x to base."""
        return self.base + x

    def reset(self) -> None:
        """Reset base."""
        self.base = 0


def test_member_filter_include_exclude():
    # Only allow 'add'
    f1 = MemberFilter(include="^add$")
    assert f1.matches("add")
    assert not f1.matches("subtract")

    # Exclude 'reset'
    f2 = MemberFilter(exclude="reset")
    assert f2.matches("add")
    assert not f2.matches("reset")

    # Both include and exclude
    f3 = MemberFilter(include=".*", exclude="^private.*")
    assert f3.matches("public_method")
    assert not f3.matches("private_method")


def test_inspect_function():
    info = inspect_function(sample_adder)
    assert info.name == "sample_adder"
    assert "a" in info.signature.parameters
    assert "b" in info.signature.parameters
    assert "Add two numbers" in (info.docstring or "")


def test_inspect_class_with_filter():
    f = MemberFilter(include="add")
    info = inspect_class(Calculator, member_filter=f)
    assert info.name == "Calculator"
    assert "add" in info.methods
    assert "reset" not in info.methods  # Filtered out


def test_import_target():
    mod = import_target("math")
    import math
    assert mod is math

    fn = import_target("math:sqrt")
    assert fn is math.sqrt
