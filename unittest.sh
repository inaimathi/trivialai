python3 -m unittest $(find tests -type f -name 'test_*.py' -exec basename {} \; | sed 's/.py$//' | sed 's/^/tests./')
