.PHONY: install run docker-build docker-run

install:
	pip install -r requirements.txt

run:
	python app.py

docker-build:
	docker build -t flask-app .

docker-run:
	docker run -p 5000:5000 flask-app
