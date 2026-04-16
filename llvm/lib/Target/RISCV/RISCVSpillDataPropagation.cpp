//===- RISCVSpillDataPropagation.cpp - Track data vregs for spills --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Track virtual registers that are known to hold plain data values so SigMode
// spills can use LD/SD. The analysis is intentionally narrow: seed from plain
// scalar loads and PseudoMovImm, then propagate only through COPY users.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-spill-data"

static cl::opt<std::string> SpillDataDumpDir(
    "riscv-spill-data-dump-dir", cl::Hidden,
    cl::desc("Directory for RISCV spill data visualization dumps"),
    cl::init("/tmp/riscv-spill-data"));

static std::string sanitizeFileName(StringRef Name) {
  std::string Result;
  Result.reserve(Name.size());
  for (char C : Name) {
    if (isAlnum(C) || C == '_' || C == '-' || C == '.')
      Result.push_back(C);
    else
      Result.push_back('_');
  }
  if (Result.empty())
    Result = "anon";
  return Result;
}

static bool getDataRegsForMI(const MachineInstr &MI,
                             const RISCVMachineFunctionInfo &RVFI,
                             SmallVectorImpl<Register> &Regs) {
  Regs.clear();
  for (const MachineOperand &MO : MI.defs()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (!Reg.isVirtual() || !RVFI.isDataValueReg(Reg))
      continue;
    Regs.push_back(Reg);
    break;
  }
  return !Regs.empty();
}

static void dumpSpillDataAnnotations(const MachineFunction &MF,
                                     const RISCVMachineFunctionInfo &RVFI) {
  std::error_code EC;
  sys::fs::create_directories(SpillDataDumpDir);

  SmallString<256> Path(SpillDataDumpDir);
  sys::path::append(Path, sanitizeFileName(MF.getName()) + ".spill-data.mir");

  raw_fd_ostream OS(Path, EC, sys::fs::OF_Text);
  if (EC) {
    LLVM_DEBUG(dbgs() << "SVT: failed to open dump file " << Path << ": "
                      << EC.message() << "\n");
    return;
  }

  OS << "MachineFunction: " << MF.getName() << "\n";
  SmallVector<Register, 8> DataRegs;
  for (const MachineBasicBlock &MBB : MF) {
    OS << "\n# " << MBB.getName() << ":\n";
    for (const MachineInstr &MI : MBB) {
      bool HasData = getDataRegsForMI(MI, RVFI, DataRegs);
      OS << (HasData ? "[DATA] " : "       ");
      MI.print(OS, /*IsStandalone=*/true);
      if (HasData) {
        OS << " ; data regs:";
        for (Register Reg : DataRegs)
          OS << ' ' << printReg(Reg);
      }
      OS << '\n';
    }
  }

  LLVM_DEBUG(dbgs() << "SVT: wrote spill data dump to " << Path << "\n");
}

namespace {
class RISCVSpillDataPropagation : public MachineFunctionPass {
public:
  static char ID;
  RISCVSpillDataPropagation() : MachineFunctionPass(ID) {
    initializeRISCVSpillDataPropagationPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "RISC-V Spill Data Propagation";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  static bool isSeedOpcode(unsigned Opcode) {
    switch (Opcode) {
    case RISCV::LB:
    case RISCV::LBU:
    case RISCV::LH:
    case RISCV::LHU:
    case RISCV::LW:
    case RISCV::LWU:
    case RISCV::LD:
    case RISCV::PseudoMovImm:
    case RISCV::SLL:
    case RISCV::SLLI:
    case RISCV::SLLW:
    case RISCV::SLLIW:
    case RISCV::SRL:
    case RISCV::SRLI:
    case RISCV::SRLW:
    case RISCV::SRLIW:
    case RISCV::SRA:
    case RISCV::SRAI:
    case RISCV::SRAW:
    case RISCV::SRAIW:
    case RISCV::AND:
    case RISCV::ANDI:
    case RISCV::OR:
    case RISCV::ORI:
    case RISCV::XOR:
    case RISCV::XORI:
    case RISCV::ADDW:
    case RISCV::ADDIW:
    case RISCV::SUBW:
    case RISCV::SLTU:
    case RISCV::SLTIU:
    case RISCV::MUL:
    case RISCV::MULH:
    case RISCV::MULHSU:
    case RISCV::MULHU:
    case RISCV::DIV:
    case RISCV::DIVU:
    case RISCV::DIVW:
    case RISCV::DIVUW:
    case RISCV::REM:
    case RISCV::REMU:
    case RISCV::REMW:
    case RISCV::REMUW:
      return true;
    default:
      return false;
    }
  }

  static Register getSingleVirtualGPRDef(const MachineInstr &MI,
                                         const MachineRegisterInfo &MRI) {
    if (MI.getNumDefs() != 1)
      return Register();
    Register DefReg = MI.getOperand(0).getReg();
    if (!DefReg.isVirtual())
      return Register();
    const TargetRegisterClass *RC = MRI.getRegClassOrNull(DefReg);
    if (!RC || !RISCV::GPRRegClass.hasSubClassEq(RC))
      return Register();
    return DefReg;
  }
};
} // namespace

char RISCVSpillDataPropagation::ID = 0;

INITIALIZE_PASS(RISCVSpillDataPropagation, "riscv-spill-data",
                "RISC-V Spill Data Propagation", false, false)

bool RISCVSpillDataPropagation::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.isSigModeSupport() || ST.getXLen() != 64)
    return false;

  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  RVFI->clearDataValueRegs();

  SmallVector<Register, 32> Worklist;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSeedOpcode(MI.getOpcode()))
        continue;
      Register DefReg = getSingleVirtualGPRDef(MI, MRI);
      if (!DefReg)
        continue;
      if (!RVFI->markDataValueReg(DefReg))
        continue;
      LLVM_DEBUG(dbgs() << "SVT: mark data reg " << printReg(DefReg)
                        << " from MI: ";
                 MI.print(dbgs()));
      Worklist.push_back(DefReg);
    }
  }

  while (!Worklist.empty()) {
    Register DataReg = Worklist.pop_back_val();
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(DataReg)) {
      if (UseMI.getOpcode() != TargetOpcode::COPY)
        continue;
      Register DefReg = getSingleVirtualGPRDef(UseMI, MRI);
      if (!DefReg)
        continue;
      if (!RVFI->markDataValueReg(DefReg))
        continue;
      LLVM_DEBUG(dbgs() << "SVT: propagate data reg " << printReg(DataReg)
                        << " -> " << printReg(DefReg) << " via MI: ";
                 UseMI.print(dbgs()));
      Worklist.push_back(DefReg);
    }
  }

  // dumpSpillDataAnnotations(MF, *RVFI);
  LLVM_DEBUG(dumpSpillDataAnnotations(MF, *RVFI));
  return false;
}

FunctionPass *llvm::createRISCVSpillDataPropagationPass() {
  return new RISCVSpillDataPropagation();
}
