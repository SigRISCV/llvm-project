// Define SDNode
def xsig_setrawid : RVSDNode<"XSIG_SETRAWID",
    SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, i64>]>, []>;
  
def riscv_rcall     : RVSDNode<"RCALL", SDT_RISCVCall,
                               [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                SDNPVariadic]>;

let Predicates = [HasVendorXSig] in {

// Define Inst
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def LS : RVInstI<0b111, OPC_CUSTOM_2, (outs GPR:$rd), (ins GPRMem:$rs1, simm12_lo:$imm12),
                 "ls", "$rd, ${imm12}(${rs1})">,
         Sched<[WriteLDW, ReadMemBase]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def SS : RVInstS<0b100, OPC_CUSTOM_2, (outs),
                 (ins GPR:$rs2, GPRMem:$rs1, simm12_lo:$imm12),
                 "ss", "$rs2, ${imm12}(${rs1})">,
         Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def LS_MAP : RVInstI<0b101, OPC_CUSTOM_2, (outs GPR:$rd), (ins GPRMem:$rs1, simm12_lo:$imm12),
                 "ls.map", "$rd, ${imm12}(${rs1})">,
         Sched<[WriteLDW, ReadMemBase]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def SS_ID : RVInstS<0b110, OPC_CUSTOM_2, (outs),
                 (ins GPR:$rs2, GPRMem:$rs1, simm12_lo:$imm12),
                 "ss.id", "$rs2, ${imm12}(${rs1})">,
         Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;

def : InstAlias<"ss.encmap ${imm12}(${rs1})", (SS_ID X0, GPRMem:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"ls.encmap ${imm12}(${rs1})", (LS_MAP X0, GPRMem:$rs1, simm12_lo:$imm12)>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def SETRAWID : RVInstI<0b000, OPC_CUSTOM_2, (outs GPR:$rd),
                     (ins GPR:$rs1, simm12_lo:$imm12),
                     "setrawid", "$rd, $rs1, $imm12">,
             Sched<[WriteIALU, ReadIALU]>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isBarrier = 1 in
def SWITCHS : RVInstR<0b0000000, 0b001, OPC_CUSTOM_2, (outs GPR:$rd),
                      (ins GPR:$rs1, GPR:$rs2),
                      "switchs", "$rd, $rs1, $rs2">;

def : InstAlias<"enterraw $rs1", (SWITCHS X0, GPR:$rs1, X0)>;
def : InstAlias<"enterraw $rs1, $rs2", (SWITCHS X0, GPR:$rs1, GPR:$rs2)>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def SETNEWID : RVInstI<0b010, OPC_CUSTOM_2, (outs GPR:$rd),
                       (ins GPR:$rs1, simm12_lo:$imm12),
                       "setnewid", "$rd, $rs1, $imm12">,
               Sched<[WriteIALU, ReadIALU]>;

let isCall = 1, Defs = [X1], isCodeGenOnly = 0, Size = 8, hasSideEffects = 0,
    mayStore = 0, mayLoad = 0 in
def PseudoRCALL : Pseudo<(outs), (ins call_symbol:$func), [],
                        "rcall", "$func">;
def PseudoRCALLIndirect : Pseudo<(outs), (ins GPRJALR:$rs1), [(riscv_rcall GPRJALR:$rs1)]>,
                         PseudoInstExpansion<(SWITCHS X1, GPR:$rs1, X0)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLSGA : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                       "lsga", "$dst, $src">;

// Define Pat
def : Pat<(xsig_setrawid GPR:$rs1), (SETRAWID GPR:$rs1, 0)>;
def : Pat<(xsig_setrawid (add GPR:$rs1, simm12_lo:$imm12)), (SETRAWID GPR:$rs1, simm12_lo:$imm12)>;
def : Pat<(riscv_rcall tglobaladdr:$func), (PseudoRCALL tglobaladdr:$func)>;
def : Pat<(riscv_rcall texternalsym:$func), (PseudoRCALL texternalsym:$func)>;
def : LdPat<sigload, LS, i64>;
def : StPat<sigstore, SS, GPR, i64>;
}