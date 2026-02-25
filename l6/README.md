
build docker image:
```sh
docker build -t shop-model:latest .
```

run docker container:
```sh
docker run -p 5000:5000 -p 9050:9050 shop-model:latest
```

- dashboard: http://localhost:9050
- api endpoint: http://localhost:5000
