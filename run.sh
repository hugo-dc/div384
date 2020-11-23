#!/bin/bash

echo Compiling huff code to evm
CODE=$(node compile.js $1)
echo Bytecode: $CODE

echo Executing evm bytecode in geth-evm
cd ./go-ethereum
./build/bin/evm --debug --statdump --code $CODE run |& tee


cd -
