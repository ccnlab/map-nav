# Makefile for Andrew Carlson's sims projects
# set STRINGERTYPES to auto generate methods for Stringer interface on an int32 datatype
EXECUTABLE_NAME=project_executable
STRINGERTYPES=Actions
# all std go defs
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

all: build

build:
	- go-bindata ./*.tsv
	- stringer -type=$(STRINGERTYPES) -output stringer.go
	$(GOBUILD) -v -o $(EXECUTABLE_NAME)
dbg-build:
	$(GOBUILD) -v -gcflags=all="-N -l" -tags debug
test: 
	$(GOTEST) -v ./...
clean:
	- rm $(EXECUTABLE_NAME)
	- rm bindata.go
	- rm stringer.go
	- $(GOCLEAN)
run:
	./$(EXECUTABLE_NAME)

# must do: go get github.com/shuLhan/go-bindata -- go install in cmd/go-bindata
bindata:
	go-bindata ./*.tsv
mac: build
linux: build
windows: build

