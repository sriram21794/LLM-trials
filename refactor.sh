black --line-length=119 $1
isort --profile black --line-length=119 $1
flake8 --max-line-length=119 $1
