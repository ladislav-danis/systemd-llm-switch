#!/bin/bash

# Ensuring that we are in the root directory of the project
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "-----------------------------------------------------------"
echo "🚀 I am running the complete SYSTEMD-LLM-SWITCH test suite."
echo "-----------------------------------------------------------"

# Run all tests in the 'tests' directory
./.venv/bin/python3 -m unittest discover -v -p "test_*.py"

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✅ All tests were successful!"
else
    echo ""
    echo "❌ Some tests failed. Check the list above."
fi

exit $RESULT
