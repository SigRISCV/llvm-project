// Define SDNode
def xsig_setrawid : RVSDNode<"XSIG_SETRAWID",
    SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, i64>]>, []>;

def xsig_setdummyid : RVSDNode<"XSIG_SETDUMMYID",
    SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, i64>]>, []>;

def xsig_setnewid : RVSDNode<"XSIG_SETNEWID",
    SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, i64>]>, []>;

def xsig_debug_sig : RVSDNode<"XSIG_DEBUG_SIG",
    SDTypeProfile<0, 0, []>, [SDNPHasChain]>;
  
def xsig_debug_chr : RVSDNode<"XSIG_DEBUG_CHR",
    SDTypeProfile<0, 1, [SDTCisVT<0, i64>]>, [SDNPHasChain]>;

def xsig_debug_int : RVSDNode<"XSIG_DEBUG_INT",
    SDTypeProfile<0, 1, [SDTCisVT<0, i64>]>, [SDNPHasChain]>;

def xsig_debug_ptr : RVSDNode<"XSIG_DEBUG_PTR",
    SDTypeProfile<0, 1, [SDTCisVT<0, i64>]>, [SDNPHasChain]>;

def xsig_debug_csr : RVSDNode<"XSIG_DEBUG_CSR",
    SDTypeProfile<0, 1, [SDTCisVT<0, i64>]>, [SDNPHasChain]>;

def xsig_debug_exit : RVSDNode<"XSIG_DEBUG_EXIT",
    SDTypeProfile<0, 0, []>, [SDNPHasChain]>;

def riscv_rcall     : RVSDNode<"RCALL", SDT_RISCVCall,
                               [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                SDNPVariadic]>;
def riscv_rtail     : RVSDNode<"RTAIL", SDT_RISCVCall,
                               [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                SDNPVariadic]>;

let Predicates = [HasVendorXSig] in {

// Define Inst
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def LS : RVInstI<0b011, OPC_MISC_MEM, (outs GPR:$rd), (ins GPRMem:$rs1, simm12_lo:$imm12),
                 "ls", "$rd, ${imm12}(${rs1})">,
         Sched<[WriteLDW, ReadMemBase]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def SS : RVInstS<0b101, OPC_STORE, (outs),
                 (ins GPR:$rs2, GPRMem:$rs1, simm12_lo:$imm12),
                 "ss", "$rs2, ${imm12}(${rs1})">,
         Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def LS_MAP : RVInstI<0b100, OPC_MISC_MEM, (outs GPR:$rd), (ins GPRMem:$rs1, simm12_lo:$imm12),
                 "ls.map", "$rd, ${imm12}(${rs1})">,
         Sched<[WriteLDW, ReadMemBase]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def SS_ID : RVInstS<0b110, OPC_STORE, (outs),
                 (ins GPR:$rs2, GPRMem:$rs1, simm12_lo:$imm12),
                 "ss.id", "$rs2, ${imm12}(${rs1})">,
         Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;

def : InstAlias<"ss.encmap ${imm12}(${rs1})", (SS_ID X0, GPRMem:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"ls.encmap ${imm12}(${rs1})", (LS_MAP X0, GPRMem:$rs1, simm12_lo:$imm12)>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def SETDUMMYID : RVInstI<0b010, OPC_CUSTOM_2, (outs GPR:$rd),
                     (ins GPR:$rs1, simm12_lo:$imm12),
                     "setdummyid", "$rd, $rs1, $imm12">,
             Sched<[WriteIALU, ReadIALU]>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def SETRAWID : RVInstI<0b110, OPC_CUSTOM_2, (outs GPR:$rd),
                     (ins GPR:$rs1, simm12_lo:$imm12),
                     "setrawid", "$rd, $rs1, $imm12">,
             Sched<[WriteIALU, ReadIALU]>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def SETNEWID : RVInstI<0b011, OPC_CUSTOM_2, (outs GPR:$rd),
                       (ins GPR:$rs1, simm12_lo:$imm12),
                       "setnewid", "$rd, $rs1, $imm12">,
               Sched<[WriteIALU, ReadIALU]>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def DEBUG : RVInstI<0b111, OPC_CUSTOM_2, (outs GPR:$rd),
                    (ins GPR:$rs1, simm12_lo:$imm12),
                    "debug", "$rd, $rs1, $imm12">,
            Sched<[]>;
def : InstAlias<"debug", (DEBUG X0, X0, 0)>;
def : InstAlias<"debug.chr $rs1", (DEBUG X0, GPR:$rs1, 1)>;
def : InstAlias<"debug.int $rs1", (DEBUG X0, GPR:$rs1, 2)>;
def : InstAlias<"debug.ptr $rs1", (DEBUG X0, GPR:$rs1, 3)>;
def : InstAlias<"debug.csr $rs1", (DEBUG X0, GPR:$rs1, 4)>;
def : InstAlias<"debug.exit", (DEBUG X0, X0, 5)>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isBarrier = 1 in
def SWITCHS : RVInstR<0b0000000, 0b100, OPC_CUSTOM_2, (outs GPR:$rd),
                      (ins GPR:$rs1, GPR:$rs2),
                      "switchs", "$rd, $rs1, $rs2">;

def : InstAlias<"enterraw $rs1", (SWITCHS X0, GPR:$rs1, X0)>;
def : InstAlias<"enterraw $rs1, $rs2", (SWITCHS X0, GPR:$rs1, GPR:$rs2)>;

let isCall = 1, Defs = [X1], isCodeGenOnly = 0, Size = 8, hasSideEffects = 0,
    mayStore = 0, mayLoad = 0 in
def PseudoRCALL : Pseudo<(outs), (ins call_symbol:$func), [],
                        "rcall", "$func">;
def PseudoRCALLIndirect : Pseudo<(outs), (ins GPRJALR:$rs1), [(riscv_rcall GPRJALR:$rs1)]>,
                         PseudoInstExpansion<(SWITCHS X1, GPR:$rs1, X0)>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [X2],
    Size = 8, isCodeGenOnly = 0, hasSideEffects = 0,
    mayStore = 0, mayLoad = 0 in
def PseudoRTAIL : Pseudo<(outs), (ins call_symbol:$dst), [],
                         "rtail", "$dst">;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [X2] in
def PseudoRTAILIndirect : Pseudo<(outs), (ins GPRTC:$rs1),
                                 [(riscv_rtail GPRTC:$rs1)]>,
                          PseudoInstExpansion<(SWITCHS X0, GPR:$rs1, X0)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLSGA : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                       "lsga", "$dst, $src">;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Size = 8, isCodeGenOnly = 1 in
def PseudoLSA_TLS_IE : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), []>;

// Define Pat
def : Pat<(xsig_setnewid GPR:$rs1), (SETNEWID GPR:$rs1, 0)>;
def : Pat<(xsig_setnewid (add GPR:$rs1, simm12_lo:$imm12)), (SETNEWID GPR:$rs1, simm12_lo:$imm12)>;
def : Pat<(xsig_setdummyid GPR:$rs1), (SETDUMMYID GPR:$rs1, 0)>;
def : Pat<(xsig_setdummyid (add GPR:$rs1, simm12_lo:$imm12)), (SETDUMMYID GPR:$rs1, simm12_lo:$imm12)>;
def : Pat<(xsig_setrawid GPR:$rs1), (SETRAWID GPR:$rs1, 0)>;
def : Pat<(xsig_setrawid (add GPR:$rs1, simm12_lo:$imm12)), (SETRAWID GPR:$rs1, simm12_lo:$imm12)>;
def : Pat<(riscv_rcall tglobaladdr:$func), (PseudoRCALL tglobaladdr:$func)>;
def : Pat<(riscv_rcall texternalsym:$func), (PseudoRCALL texternalsym:$func)>;
def : Pat<(riscv_rtail tglobaladdr:$func), (PseudoRTAIL tglobaladdr:$func)>;
def : Pat<(riscv_rtail texternalsym:$func), (PseudoRTAIL texternalsym:$func)>;
def : LdPat<sigload, LS, i64>;
def : StPat<sigstore, SS, GPR, i64>;
def : LdPat<dynsigload, LS_MAP, i64>;
def : StPat<dynsigstore, SS_ID, GPR, i64>;
def : Pat<(xsig_debug_sig), (DEBUG (XLenVT X0), 0)>;
def : Pat<(xsig_debug_chr GPR:$rs1), (DEBUG GPR:$rs1, 1)>;
def : Pat<(xsig_debug_int GPR:$rs1), (DEBUG GPR:$rs1, 2)>;
def : Pat<(xsig_debug_ptr GPR:$rs1), (DEBUG GPR:$rs1, 3)>;
def : Pat<(xsig_debug_csr GPR:$rs1), (DEBUG GPR:$rs1, 4)>;
def : Pat<(xsig_debug_exit), (DEBUG (XLenVT X0), 5)>;
}
