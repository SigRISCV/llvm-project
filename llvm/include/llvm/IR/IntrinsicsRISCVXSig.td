//===- IntrinsicsRISCVXCV.td - CORE-V intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Sig vendor intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//


let TargetPrefix = "riscv" in {

def int_riscv_xsig_setrawid : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem, IntrWillReturn]>,
  ClangBuiltin<"__builtin_riscv_xsig_setrawid">;

def int_riscv_xsig_setnewid : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem, IntrWillReturn]>,
  ClangBuiltin<"__builtin_riscv_xsig_setnewid">;

} // TargetPrefix = "riscv"
