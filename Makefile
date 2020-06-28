.PHONY: build

NAME_DEV   := explore/surprise
# TAG_DEV    := $$(git rev-parse --short HEAD)
IMG_DEV    := ${NAME_DEV}:0.1
LATEST_DEV := ${NAME_DEV}:latest

build:
	@docker build -t ${IMG_DEV} -f Dockerfile .
	@docker tag ${IMG_DEV} ${LATEST_DEV}

run:
	@docker run -p "8880:8888" --name explore_surprise -v `pwd`:/proj ${NAME_DEV}:latest

remove:
	@docker rm -f explore_surprise
