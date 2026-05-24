.PHONY: build fmt test clean

build:
	go build ./cmd/llama-cpp-perfkit

fmt:
	go fmt ./...

test:
	go test ./...

clean:
	rm -f llama-cpp-perfkit
