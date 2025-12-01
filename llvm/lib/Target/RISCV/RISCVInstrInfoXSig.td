class BareSImm12PcRelMaybeSym : Operand<OtherVT> {
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<12>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

def simm12_pcrel : BareSImm12PcRelMaybeSym,
                   ImmLeaf<XLenVT, [{return isInt<12>(Imm);}]> {
  let ParserMatchClass = BareSImmNAsmOperand<12>;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<12>";
  let OperandType = "OPERAND_PCREL";
}

let Predicates = [HasVendorXSig] in {

// ls uses custom opcode OPC_CUSTOM_0 (0b0001011) with I-type format
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def LS : RVInstI<0b111, OPC_CUSTOM_0, (outs GPR:$rd), (ins GPRMem:$rs1, simm12_lo:$imm12),
                 "ls", "$rd, ${imm12}(${rs1})">,
         Sched<[WriteLDW, ReadMemBase]>;
// ss uses custom opcode OPC_CUSTOM_1 (0b0101011) with S-type format
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def SS : RVInstS<0b100, OPC_CUSTOM_1, (outs),
                 (ins GPR:$rs2, GPRMem:$rs1, simm12_lo:$imm12),
                 "ss", "$rs2, ${imm12}(${rs1})">,
         Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def SETRAW : RVInstI<0b000, OPC_CUSTOM_2, (outs GPR:$rd),
                     (ins GPR:$rs1, simm12_lo:$imm12),
                     "setraw", "$rd, $rs1, $imm12">,
             Sched<[WriteIALU, ReadIALU]>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isBarrier = 1, isTerminator = 1 in
def SWITCHS : RVInstI<0b001, OPC_CUSTOM_2, (outs),
                      (ins GPR:$rs1, simm12_pcrel:$imm12),
                      "switchs", "$rs1, $imm12">,
              Sched<[WriteJmp]> {
  let rd = 0;
}

def : InstAlias<"switchs $imm12", (SWITCHS X0, simm12_pcrel:$imm12), 4>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def SETNEWID : RVInstI<0b010, OPC_CUSTOM_2, (outs GPR:$rd),
                       (ins GPR:$rs1, simm12_lo:$imm12),
                       "setnewid", "$rd, $rs1, $imm12">,
               Sched<[WriteIALU, ReadIALU]>;

}