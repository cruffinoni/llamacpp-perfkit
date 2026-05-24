.PHONY: build fmt test clean dev


build:
	go build -o llama-cpp-perfkit ./cmd/llama-cpp-perfkit

dev: build
	./llama-cpp-perfkit dev tui

fmt:
	go fmt ./...

test:
	go test ./...

clean:
	rm -f llama-cpp-perfkit
