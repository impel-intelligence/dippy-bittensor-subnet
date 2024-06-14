.PHONY: install lint test clean

# Define variables
PACKAGE_NAME=dippy-bittensor-subnet
TEST_PATH=./tests

# Install dependencies
install:
	pip install -r requirements.txt

# Lint the project
lint:
	flake8 $(PACKAGE_NAME)

# Run tests with pytest
test:
	pytest $(TEST_PATH)

# Clean up pyc files and __pycache__ directories
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +