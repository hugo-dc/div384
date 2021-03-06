const path = require('path');
const fs = require('fs');

const { compiler } = require('./huff/src');
const parser = require('./huff/src/parser');

const pathToData = path.posix.resolve(__dirname, './');

const huff_file = process.argv[2]

const { inputMap, macros, jumptables } = parser.parseFile(huff_file+".huff", pathToData);

const {
    data: { bytecode: macroCode },
//} = parser.processMacro('MILLER_LOOP_TEST_HARDCODED', 0, [], macros, inputMap, jumptables);
} = parser.processMacro(huff_file.toUpperCase(), 0, [], macros, inputMap, jumptables);

console.log("0x"+macroCode)

