# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xsig -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-xsig < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xsig -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xsig -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-xsig < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xsig -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s

# CHECK-ASM: calleesave 28, 496(sp)
# CHECK-ASM: encoding: [0x5b,0x0e,0x01,0x1f]
# CHECK-OBJ: calleesave 0x1c, 0x1f0(sp)
calleesave 28, 0x1f0(sp)

# CHECK-ASM: calleerestore 28, 496(sp)
# CHECK-ASM: encoding: [0x5b,0x1e,0x01,0x1f]
# CHECK-OBJ: calleerestore 0x1c, 0x1f0(sp)
calleerestore 28, 0x1f0(sp)

# CHECK-ASM: calleesave 4, -16(s0)
# CHECK-ASM: encoding: [0x5b,0x02,0x04,0xff]
# CHECK-OBJ: calleesave 0x4, -0x10(s0)
calleesave 4, -16(s0)

# CHECK-ASM: calleerestore 17, 2047(t0)
# CHECK-ASM: encoding: [0xdb,0x98,0xf2,0x7f]
# CHECK-OBJ: calleerestore 0x11, 0x7ff(t0)
calleerestore 17, 2047(t0)
