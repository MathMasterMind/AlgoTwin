# Description: Extracts P-Code from all functions in the current program and saves it as a JSON file.

from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.listing import FunctionManager
import json

def get_pcode_for_functions():
    currentProgram = getCurrentProgram()
    fm = currentProgram.getFunctionManager()
    decomp = DecompInterface()
    decomp.openProgram(currentProgram)

    functions_pcode = {}

    for func in fm.getFunctions(True):
        results = decomp.decompileFunction(func, 60, None)
        if results.decompileCompleted():
            pcode_ops = []
            for op in results.getHighFunction().getPcodeOps():
                pcode_ops.append(str(op))
            functions_pcode[func.getName()] = pcode_ops

    # Save as JSON for further processing
    with open("/path/to/save/pcode.json", "w") as f:
        json.dump(functions_pcode, f, indent=2)

    print("P-Code extraction complete!")

get_pcode_for_functions()
    

