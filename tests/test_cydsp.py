"""Tests for cydsp nanobind extension module."""

import cydsp


def test_add():
    """Test add function."""
    assert cydsp.add(1, 2) == 3
    assert cydsp.add(-1, 1) == 0
    assert cydsp.add(0, 0) == 0


def test_greet():
    """Test greet function."""
    assert cydsp.greet("World") == "Hello, World!"
    assert cydsp.greet("Python") == "Hello, Python!"
