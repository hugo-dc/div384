const path = require('path');
const fs = require('fs');

const { compiler } = require('./huff/src');
const parser = require('./huff/src/parser');

const pathToData = path.posix.resolve(__dirname, './');

const { inputMap, macros, jumptables } = parser.parseFile('div384.huff', pathToData);

const {
    data: { bytecode: macroCode },
//} = parser.processMacro('MILLER_LOOP_TEST_HARDCODED', 0, [], macros, inputMap, jumptables);
} = parser.processMacro('DIV384', 0, [], macros, inputMap, jumptables);

console.log("0x"+macroCode)

