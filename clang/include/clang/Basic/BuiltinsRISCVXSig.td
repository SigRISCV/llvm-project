//==- BuiltinsRISCVXAndes.td - RISC-V Andes Builtin database -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Andes-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

class RISCVXSigBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_xsig_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// XSig extension.
//===----------------------------------------------------------------------===//

def setrawid : RISCVXSigBuiltin<"void*(void*)", "experimental-xsig">;
def setnewid : RISCVXSigBuiltin<"void*(void*)", "experimental-xsig">;
def setdummyid : RISCVXSigBuiltin<"void*(void*)", "experimental-xsig">;
def load_sig : RISCVXSigBuiltin<"void*(void*)", "experimental-xsig">;
def store_sig : RISCVXSigBuiltin<"void(void*, void*)", "experimental-xsig">;
def load_dyn : RISCVXSigBuiltin<"void*(void*)", "experimental-xsig">;
def store_dyn : RISCVXSigBuiltin<"void(void*, void*)", "experimental-xsig">;
def debug_sig : RISCVXSigBuiltin<"void()", "experimental-xsig">;
def debug_chr : RISCVXSigBuiltin<"void(uint64_t)", "experimental-xsig">;
def debug_int : RISCVXSigBuiltin<"void(uint64_t)", "experimental-xsig">;
def debug_ptr : RISCVXSigBuiltin<"void(void*)", "experimental-xsig">;
def debug_csr : RISCVXSigBuiltin<"void(uint64_t)", "experimental-xsig">;
} // Attributes = [NoThrow, Const]
