flake8 := flake8 .
mypy := mypy .

# to run program in docker
run:
	docker-compose down
	docker-compose build
	docker-compose up
# for Docker compose v2
run-v2:
	docker compose build
	docker compose up
# mypy is a static type checker
mypy:
	$(mypy)

test:
	pytest -v --cov . --cov-report term-missing \
	--cov-fail-under=100 -n 4 --reuse-db -W error \
	-W ignore::ResourceWarning

outdated:
	pip list --outdated --format=columns

flake8:
	$(flake8)

safety:
	$(safety)

lint:
	$(flake8) && $(mypy)

check:
	python manage.py check --deploy

warnings:
	python -Wd manage.py check
