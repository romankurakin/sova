# Search questions

187 English queries; editorial revision 7. The editor checked every source reference; independent review is pending.

Generated from [queries.json](queries.json). References use 1-based PDF pages; they are authoring aids, not relevance labels or exclusive answer locations.

20 translations included by default; editorial revision 6. Generated from [language-probes.json](language-probes.json) and shown below their English anchors. They share the anchor's intent and are scored separately by language. Independent translation review is pending; this does not disable any queries.

## Instructions, atomics and memory ordering

### q001 — writing to a zero register

Intent: Understand the effect of an instruction writing to the constant-zero register.

Category: `conceptual` · ISA: `both` · Comparison group: `zero_register`

- [riscv-unprivileged, PDF p. 43](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=43) — 2.1. Programmers' Model for Base Integer ISA
- [loongarch-vol1, PDF p. 20](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=20) — 2.1.2.1. General-purpose Registers

### q002 — What happens when a RISC-V instruction writes to x0?

Intent: Understand the effect of an instruction writing to the constant-zero register.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `zero_register`

- [riscv-unprivileged, PDF p. 43](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=43) — 2.1. Programmers' Model for Base Integer ISA

### q003 — What happens when a LoongArch instruction writes to r0?

Intent: Understand the effect of an instruction writing to the constant-zero register.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `zero_register`

- [loongarch-vol1, PDF p. 20](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=20) — 2.1.2.1. General-purpose Registers

### q004 — signed versus unsigned byte loads

Intent: Compare how signed and unsigned byte loads extend the value to the register width.

Category: `comparison` · ISA: `both` · Comparison group: `load_extension`

- [riscv-unprivileged, PDF p. 54](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=54) — 2.6. Load and Store Instructions
- [loongarch-vol1, PDF p. 50](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=50) — 2.2.5.1. LD.{B[U]/H[U]/W[U]/D}, ST.{B/H/W/D}
- [loongarch-vol1, PDF p. 51](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=51) — 2.2.5.1. LD.{B[U]/H[U]/W[U]/D}, ST.{B/H/W/D}

### q005 — In RISC-V, how do LB and LBU differ when the loaded byte has its top bit set?

Intent: Compare how signed and unsigned byte loads extend the value to the register width.

Category: `comparison` · ISA: `riscv` · Comparison group: `load_extension`

- [riscv-unprivileged, PDF p. 54](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=54) — 2.6. Load and Store Instructions

#### probe-ru-q005 (RU)

Чем в RISC-V отличаются LB и LBU, если старший бит загруженного байта равен 1?

#### probe-zh-q005 (ZH)

在 RISC-V 中，加载的字节最高位为 1 时，LB 和 LBU 有什么区别？

### q006 — In LoongArch, how do LD.B and LD.BU differ when the loaded byte has its top bit set?

Intent: Compare how signed and unsigned byte loads extend the value to the register width.

Category: `comparison` · ISA: `loongarch` · Comparison group: `load_extension`

- [loongarch-vol1, PDF p. 50](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=50) — 2.2.5.1. LD.{B[U]/H[U]/W[U]/D}, ST.{B/H/W/D}
- [loongarch-vol1, PDF p. 51](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=51) — 2.2.5.1. LD.{B[U]/H[U]/W[U]/D}, ST.{B/H/W/D}

### q007 — branch conditions and target address calculation

Intent: Understand a conditional branch and its destination-address calculation.

Category: `conceptual` · ISA: `both` · Comparison group: `conditional_branch`

- [riscv-unprivileged, PDF p. 52](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=52) — 2.5.2. Conditional Branches
- [loongarch-vol1, PDF p. 47](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=47) — 2.2.4.1. BEQ, BNE, BLT[U], BGE[U]
- [loongarch-vol1, PDF p. 46](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=46) — 2.2.4.1. BEQ, BNE, BLT[U], BGE[U]

### q008 — How is the destination address calculated for a taken RISC-V BEQ?

Intent: Determine the destination-address calculation for a taken RISC-V BEQ.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `conditional_branch`

- [riscv-unprivileged, PDF p. 52](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=52) — 2.5.2. Conditional Branches

### q009 — How is the destination address calculated for a taken LoongArch BEQ?

Intent: Determine the destination-address calculation for a taken LoongArch BEQ.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `conditional_branch`

- [loongarch-vol1, PDF p. 47](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=47) — 2.2.4.1. BEQ, BNE, BLT[U], BGE[U]
- [loongarch-vol1, PDF p. 46](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=46) — 2.2.4.1. BEQ, BNE, BLT[U], BGE[U]

### q010 — signed and unsigned integer comparison instructions

Intent: Compare signed and unsigned integer comparison instructions.

Category: `comparison` · ISA: `both` · Comparison group: `signed_compare`

- [riscv-unprivileged, PDF p. 49](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=49) — 2.4.2. Integer Register-Register Operations
- [loongarch-vol1, PDF p. 28](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=28) — 2.2.1.5. SLT[U]

### q011 — How do SLT and SLTU differ in RISC-V?

Intent: Compare signed and unsigned integer comparison instructions.

Category: `comparison` · ISA: `riscv` · Comparison group: `signed_compare`

- [riscv-unprivileged, PDF p. 49](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=49) — 2.4.2. Integer Register-Register Operations

### q012 — How do SLT and SLTU differ in LoongArch?

Intent: Compare signed and unsigned integer comparison instructions.

Category: `comparison` · ISA: `loongarch` · Comparison group: `signed_compare`

- [loongarch-vol1, PDF p. 28](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=28) — 2.2.1.5. SLT[U]

### q013 — high half of an integer multiplication result

Intent: Determine the operand interpretation and result of high-half integer multiplication.

Category: `conceptual` · ISA: `both` · Comparison group: `multiply_high`

- [riscv-unprivileged, PDF p. 89](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=89) — 12.1. Multiplication Operations
- [loongarch-vol1, PDF p. 33](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=33) — 2.2.1.11. MUL.{W/D}, MULH, {W[U]/D[U]}

### q014 — How do RISC-V MULH, MULHU and MULHSU interpret their operands?

Intent: Determine the operand interpretation and result of high-half integer multiplication.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `multiply_high`

- [riscv-unprivileged, PDF p. 89](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=89) — 12.1. Multiplication Operations

### q015 — How do LoongArch MULH.D and MULH.DU interpret their operands?

Intent: Determine the operand interpretation and result of high-half integer multiplication.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `multiply_high`

- [loongarch-vol1, PDF p. 33](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=33) — 2.2.1.11. MUL.{W/D}, MULH, {W[U]/D[U]}

### q016 — integer division by zero: result or exception?

Intent: Determine the result and exception behavior of integer division by zero.

Category: `conceptual` · ISA: `both` · Comparison group: `division_edge`

- [riscv-unprivileged, PDF p. 90](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=90) — 12.2. Division Operations
- [loongarch-vol1, PDF p. 35](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=35) — 2.2.1.13. DIV.{W[U]/D[U]}, MOD.{W[U]/D[U]}

### q017 — What are the RISC-V DIV and REM results for division by zero?

Intent: Determine the result and exception behavior of integer division by zero.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `division_edge`

- [riscv-unprivileged, PDF p. 90](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=90) — 12.2. Division Operations

### q018 — What does LoongArch specify for integer division with a zero divisor?

Intent: Determine the result and exception behavior of integer division by zero.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `division_edge`

- [loongarch-vol1, PDF p. 35](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=35) — 2.2.1.13. DIV.{W[U]/D[U]}, MOD.{W[U]/D[U]}

### q019 — conditional store keeps failing in an atomic update loop

Intent: Diagnose repeated conditional-store failure in an atomic-update loop.

Category: `problem_description` · ISA: `both` · Comparison group: `conditional_store`

- [riscv-unprivileged, PDF p. 93](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=93) — 13.2. Zalrsc Extension for Load-Reserved/Store-Conditional Instructions
- [riscv-unprivileged, PDF p. 94](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=94) — 13.2. Zalrsc Extension for Load-Reserved/Store-Conditional Instructions
- [riscv-unprivileged, PDF p. 96](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=96) — 13.3. Eventual Success of Store-Conditional Instructions
- [loongarch-vol1, PDF p. 65](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=65) — 2.2.7.4. LL.{W/D}, SC.{W/D}

### q020 — Why can a RISC-V SC.W fail after LR.W, and what does its result indicate?

Intent: Understand conditional-store failures and the meaning of their return status in an atomic-update loop.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `conditional_store`

- [riscv-unprivileged, PDF p. 93](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=93) — 13.2. Zalrsc Extension for Load-Reserved/Store-Conditional Instructions
- [riscv-unprivileged, PDF p. 94](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=94) — 13.2. Zalrsc Extension for Load-Reserved/Store-Conditional Instructions
- [riscv-unprivileged, PDF p. 96](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=96) — 13.3. Eventual Success of Store-Conditional Instructions

### q021 — Why can a LoongArch SC.W fail after LL.W, and what does its result indicate?

Intent: Understand conditional-store failures and the meaning of their return status in an atomic-update loop.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `conditional_store`

- [loongarch-vol1, PDF p. 65](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=65) — 2.2.7.4. LL.{W/D}, SC.{W/D}

#### probe-ru-q021 (RU)

Почему в LoongArch SC.W может завершиться неудачей после LL.W и что означает возвращаемое значение?

#### probe-zh-q021 (ZH)

为什么 LoongArch 的 SC.W 在 LL.W 之后仍可能失败？返回结果表示什么？

### q022 — atomic counter increment returning previous value

Intent: Find an atomic memory increment that also returns the previous value.

Category: `exact_lookup` · ISA: `both` · Comparison group: `atomic_add`

- [riscv-unprivileged, PDF p. 97](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=97) — 13.4. Zaamo Extension for Atomic Memory Operations
- [riscv-unprivileged, PDF p. 98](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=98) — 13.4. Zaamo Extension for Atomic Memory Operations
- [loongarch-vol1, PDF p. 62](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=62) — 2.2.7.1. Atomic memory access instructions

### q023 — Which RISC-V instruction atomically adds to memory and returns the old value?

Intent: Find an atomic memory increment that also returns the previous value.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `atomic_add`

- [riscv-unprivileged, PDF p. 97](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=97) — 13.4. Zaamo Extension for Atomic Memory Operations
- [riscv-unprivileged, PDF p. 98](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=98) — 13.4. Zaamo Extension for Atomic Memory Operations

### q024 — Which LoongArch instruction atomically adds to memory and returns the old value?

Intent: Find an atomic memory increment that also returns the previous value.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `atomic_add`

- [loongarch-vol1, PDF p. 62](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=62) — 2.2.7.1. Atomic memory access instructions

### q025 — memory ordering barriers

Intent: Determine memory-ordering guarantees of a barrier and its control fields.

Category: `conceptual` · ISA: `both` · Comparison group: `memory_barrier`

- [riscv-unprivileged, PDF p. 56](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=56) — 2.7. Memory Ordering Instructions
- [loongarch-vol1, PDF p. 66](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=66) — 2.2.8.1. DBAR

### q026 — In RISC-V, what do the predecessor and successor fields of FENCE select?

Intent: Interpret the predecessor and successor fields in a RISC-V FENCE instruction.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `memory_barrier`

- [riscv-unprivileged, PDF p. 56](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=56) — 2.7. Memory Ordering Instructions

### q027 — What memory ordering does LoongArch DBAR 0 guarantee?

Intent: Determine memory-ordering guarantees of a barrier and its control fields.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `memory_barrier`

- [loongarch-vol1, PDF p. 66](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=66) — 2.2.8.1. DBAR

### q028 — instruction cache synchronization after modifying code

Intent: Find the synchronization needed before fetching modified instructions.

Category: `procedure` · ISA: `both` · Comparison group: `instruction_sync`

- [riscv-unprivileged, PDF p. 68](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=68) — Chapter 5. Zifencei Extension for Instruction-Fetch Fence, Version 2.0
- [loongarch-vol1, PDF p. 67](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=67) — 2.2.8.2. IBAR

### q029 — When does RISC-V require FENCE.I after writing executable code?

Intent: Find the synchronization needed before fetching modified instructions.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `instruction_sync`

- [riscv-unprivileged, PDF p. 68](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=68) — Chapter 5. Zifencei Extension for Instruction-Fetch Fence, Version 2.0

### q030 — What is the role of IBAR in LoongArch instruction-fetch synchronization?

Intent: Find the synchronization needed before fetching modified instructions.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `instruction_sync`

- [loongarch-vol1, PDF p. 67](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=67) — 2.2.8.2. IBAR

### q031 — Which instruction transfers control to a handler for an operating-system service request?

Intent: Find the instruction mechanism for requesting operating-system services.

Category: `exact_lookup` · ISA: `both` · Comparison group: `system_call`

- [riscv-unprivileged, PDF p. 57](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=57) — 2.8. Environment Call and Breakpoints
- [riscv-privileged, PDF p. 64](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=64) — 3.3.1. Environment Call and Breakpoint
- [loongarch-vol1, PDF p. 68](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=68) — 2.2.10.1. syscall

### q032 — What exception does RISC-V ECALL cause, and how does it depend on the current privilege mode?

Intent: Determine how the originating privilege mode affects the exception raised by ECALL.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `system_call`

- [riscv-unprivileged, PDF p. 57](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=57) — 2.8. Environment Call and Breakpoints
- [riscv-privileged, PDF p. 64](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=64) — 3.3.1. Environment Call and Breakpoint

### q033 — How does the LoongArch SYSCALL instruction invoke exception handling?

Intent: Understand the exception raised by the LoongArch SYSCALL instruction.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `system_call`

- [loongarch-vol1, PDF p. 68](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=68) — 2.2.10.1. syscall

### q034 — software breakpoint instruction

Intent: Understand instruction support for software breakpoints.

Category: `exact_lookup` · ISA: `both` · Comparison group: `breakpoint`

- [riscv-privileged, PDF p. 64](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=64) — 3.3.1. Environment Call and Breakpoint
- [loongarch-vol1, PDF p. 68](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=68) — 2.2.10.2. break

### q035 — What does EBREAK do in RISC-V?

Intent: Understand instruction support for software breakpoints.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `breakpoint`

- [riscv-privileged, PDF p. 64](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=64) — 3.3.1. Environment Call and Breakpoint

### q036 — What does BREAK do in LoongArch?

Intent: Understand instruction support for software breakpoints.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `breakpoint`

- [loongarch-vol1, PDF p. 68](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=68) — 2.2.10.2. break

### q037 — floating-point rounding mode: global setting and instruction-specific control

Intent: Determine the relationship between global and instruction-specific floating-point rounding selection.

Category: `conceptual` · ISA: `both` · Comparison group: `fp_rounding`

- [riscv-unprivileged, PDF p. 141](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=141) — 20.2. Floating-Point Control and Status Register
- [loongarch-vol1, PDF p. 79](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=79) — 3.1.3.3. Floating-point Control and Status Register
- [loongarch-vol1, PDF p. 95](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=95) — 3.2.3.3. FTINT{RM/RP/RZ/RNE}.{W/L}.{S/D}

### q038 — How do the rm field and frm register select rounding mode in RISC-V?

Intent: Determine the relationship between global and instruction-specific floating-point rounding selection.

Category: `comparison` · ISA: `riscv` · Comparison group: `fp_rounding`

- [riscv-unprivileged, PDF p. 141](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=141) — 20.2. Floating-Point Control and Status Register

### q039 — How is floating-point rounding mode selected through LoongArch FCSR and conversion instructions?

Intent: Determine the relationship between global and instruction-specific floating-point rounding selection.

Category: `comparison` · ISA: `loongarch` · Comparison group: `fp_rounding`

- [loongarch-vol1, PDF p. 79](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=79) — 3.1.3.3. Floating-point Control and Status Register
- [loongarch-vol1, PDF p. 95](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=95) — 3.2.3.3. FTINT{RM/RP/RZ/RNE}.{W/L}.{S/D}

### q040 — How do floating-point comparisons behave when an operand is NaN?

Intent: Understand comparison outcomes and exception behavior when an operand is NaN.

Category: `conceptual` · ISA: `both` · Comparison group: `fp_nan`

- [riscv-unprivileged, PDF p. 146](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=146) — 20.8. Single-Precision Floating-Point Compare Instructions
- [loongarch-vol1, PDF p. 92](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=92) — 3.2.2.1. FCMP.cond.{S/D}

### q041 — How do RISC-V FEQ, FLT and FLE handle NaN operands and exception flags?

Intent: Understand comparison outcomes and exception behavior when an operand is NaN.

Category: `comparison` · ISA: `riscv` · Comparison group: `fp_nan`

- [riscv-unprivileged, PDF p. 146](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=146) — 20.8. Single-Precision Floating-Point Compare Instructions

### q042 — How do LoongArch FCMP conditions distinguish ordered and unordered comparisons?

Intent: Understand comparison outcomes and exception behavior when an operand is NaN.

Category: `comparison` · ISA: `loongarch` · Comparison group: `fp_nan`

- [loongarch-vol1, PDF p. 92](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=92) — 3.2.2.1. FCMP.cond.{S/D}

### q043 — Where do floating-point-to-integer conversions put the integer result in RISC-V and LoongArch?

Intent: Compare the destination register files of RISC-V and LoongArch floating-point-to-integer conversion instructions.

Category: `comparison` · ISA: `both` · Comparison group: `fp_conversion`

- [riscv-unprivileged, PDF p. 145](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=145) — 20.7. Single-Precision Floating-Point Conversion and Move Instructions
- [loongarch-vol1, PDF p. 94](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=94) — 3.2.3.2. FFINT{S/D}.{W/L}, FTINT.{W/L}.{S/D}

### q044 — What result and flags does RISC-V FCVT.W.S produce for an out-of-range value?

Intent: Determine FCVT.W.S results and flags for inputs outside the destination integer range.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `fp_conversion`

- [riscv-unprivileged, PDF p. 145](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=145) — 20.7. Single-Precision Floating-Point Conversion and Move Instructions

### q045 — How can I extract a bit field from a LoongArch register without a shift-and-mask sequence?

Intent: Find a LoongArch instruction for extracting a contiguous bit range from a general register.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `bitfield_extract`

- [loongarch-vol1, PDF p. 45](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=45) — 2.2.3.9. BSTRPICK.{W/D}

### q046 — move raw bits between integer and floating-point registers

Intent: Find instructions that move floating-point bit representations between integer and floating-point registers.

Category: `exact_lookup` · ISA: `both` · Comparison group: `fp_bit_move`

- [riscv-unprivileged, PDF p. 146](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=146) — 20.7. Single-Precision Floating-Point Conversion and Move Instructions
- [loongarch-vol1, PDF p. 100](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=100) — 3.2.4.4. MOVFR2GR.{S/D}, MOVFRH2GR.S

### q047 — How does RISC-V FMV.X.W differ from FCVT.W.S?

Intent: Distinguish representation transfer from numeric conversion for RISC-V FMV.X.W and FCVT.W.S.

Category: `comparison` · ISA: `riscv` · Comparison group: `fp_bit_move`

- [riscv-unprivileged, PDF p. 146](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=146) — 20.7. Single-Precision Floating-Point Conversion and Move Instructions
- [riscv-unprivileged, PDF p. 145](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=145) — 20.7. Single-Precision Floating-Point Conversion and Move Instructions

### q048 — How does LoongArch MOVFR2GR.S differ from FTINT.W.S?

Intent: Distinguish representation transfer from numeric conversion for LoongArch MOVFR2GR.S and FTINT.W.S.

Category: `comparison` · ISA: `loongarch` · Comparison group: `fp_bit_move`

- [loongarch-vol1, PDF p. 100](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=100) — 3.2.4.4. MOVFR2GR.{S/D}, MOVFRH2GR.S
- [loongarch-vol1, PDF p. 94](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=94) — 3.2.3.2. FFINT{S/D}.{W/L}, FTINT.{W/L}.{S/D}

### q049 — change selected control register bits without overwriting other fields

Intent: Find how to update selected CSR fields while preserving other fields.

Category: `conceptual` · ISA: `both` · Comparison group: `csr_update`

- [riscv-unprivileged, PDF p. 69](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=69) — 6.1. CSR Instructions
- [loongarch-vol1, PDF p. 108](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=108) — 4.2.1. CSR Access Instructions

### q050 — How do CSRRS and CSRRC update a RISC-V CSR?

Intent: Find how to update selected CSR fields while preserving other fields.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `csr_update`

- [riscv-unprivileged, PDF p. 69](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=69) — 6.1. CSR Instructions

### q051 — How does CSRXCHG update selected bits of a LoongArch CSR?

Intent: Find how to update selected CSR fields while preserving other fields.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `csr_update`

- [loongarch-vol1, PDF p. 108](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=108) — 4.2.1. CSR Access Instructions

### q052 — What should software do with reserved or unsupported fields in control registers?

Intent: Determine safe software treatment of reserved or unsupported control-register fields.

Category: `conceptual` · ISA: `both` · Comparison group: `reserved_fields`

- [riscv-privileged, PDF p. 28](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=28) — 2.3.1. Reserved Writes Preserve Values, Reads Ignore Values (WPRI)
- [riscv-privileged, PDF p. 28](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=28) — 2.3.2. Write/Read Only Legal Values (WLRL)
- [riscv-privileged, PDF p. 28](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=28) — 2.3.3. Write Any Values, Reads Legal Values (WARL)
- [loongarch-vol1, PDF p. 135](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=135) — 7.2.1. Attributes of Reading and Writing

### q053 — What do WPRI, WLRL and WARL mean for RISC-V CSR fields?

Intent: Distinguish the software obligations and read/write guarantees of WPRI, WLRL, and WARL CSR fields.

Category: `comparison` · ISA: `riscv` · Comparison group: `reserved_fields`

- [riscv-privileged, PDF p. 28](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=28) — 2.3.1. Reserved Writes Preserve Values, Reads Ignore Values (WPRI)
- [riscv-privileged, PDF p. 28](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=28) — 2.3.2. Write/Read Only Legal Values (WLRL)
- [riscv-privileged, PDF p. 28](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=28) — 2.3.3. Write Any Values, Reads Legal Values (WARL)

### q054 — Is it safe for portable LoongArch software to write an unimplemented CSR?

Intent: Determine whether writing an unimplemented LoongArch CSR is safe for software compatibility.

Category: `conceptual` · ISA: `loongarch` · Comparison group: `reserved_fields`

- [loongarch-vol1, PDF p. 136](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=136) — 7.2.3. Access Effects of Undefined and Unimplemented Control and Status Registers

### q091 — How do RISC-V and LoongArch specify byte order for memory accesses?

Intent: Compare the memory byte-order rules of RISC-V and LoongArch.

Category: `comparison` · ISA: `both` · Comparison group: `endianness`

- [riscv-unprivileged, PDF p. 54](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=54) — 2.6. Load and Store Instructions
- [loongarch-vol1, PDF p. 22](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=22) — 2.1.6. Endian

### q092 — What can happen when a load or store address is not naturally aligned?

Intent: Determine possible outcomes when a load or store address is not naturally aligned.

Category: `conceptual` · ISA: `both` · Comparison group: `unaligned`

- [riscv-unprivileged, PDF p. 55](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=55) — 2.6. Load and Store Instructions
- [loongarch-vol1, PDF p. 23](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=23) — 2.1.8. Unaligned Memory Access

### q093 — discover processor identity and supported instruction sets

Intent: Find architectural facilities for discovering processor identity and supported instructions.

Category: `conceptual` · ISA: `both` · Comparison group: `cpu_identity`

- [riscv-privileged, PDF p. 35](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=35) — 3.1.3. Machine Architecture ID (marchid) Register
- [riscv-privileged, PDF p. 31](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=31) — 3.1.1. Machine ISA (misa) Register
- [loongarch-vol1, PDF p. 70](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=70) — 2.2.10.5. cpucfg
- [loongarch-vol1, PDF p. 69](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=69) — 2.2.10.5. cpucfg

### q095 — How are floating-point exception flags recorded and cleared?

Intent: Understand how software reads and clears accumulated floating-point exception status.

Category: `conceptual` · ISA: `both` · Comparison group: `fp_flags`

- [riscv-unprivileged, PDF p. 141](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=141) — 20.2. Floating-Point Control and Status Register
- [riscv-unprivileged, PDF p. 140](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=140) — 20.2. Floating-Point Control and Status Register
- [loongarch-vol1, PDF p. 80](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=80) — 3.1.3.3. Floating-point Control and Status Register
- [loongarch-vol1, PDF p. 100](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=100) — 3.2.4.5. MOVGR2FCSR, MOVFCSR2GR

### q097 — How can software select and count processor performance events?

Intent: Find how software configures and reads hardware performance-event counters.

Category: `procedure` · ISA: `both` · Comparison group: `counters`

- [riscv-privileged, PDF p. 51](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=51) — 3.1.10. Hardware Performance Monitor
- [loongarch-vol1, PDF p. 165](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=165) — 7.8.1. Performance Monitor Configuration n (PMCFG)
- [loongarch-vol1, PDF p. 166](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=166) — 7.8.2. Performance Monitor Overall Counter n (PMCNT)

### q107 — Does a local FENCE.I synchronize instruction fetching on every RISC-V hart?

Intent: Determine whether a local instruction-fetch fence synchronizes other RISC-V harts.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `instruction_sync`

- [riscv-unprivileged, PDF p. 68](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=68) — Chapter 5. Zifencei Extension for Instruction-Fetch Fence, Version 2.0

#### probe-ru-q107 (RU)

Синхронизирует ли локальная инструкция FENCE.I выборку инструкций на всех hart RISC-V?

#### probe-zh-q107 (ZH)

本地执行 FENCE.I 会同步所有 RISC-V hart 的取指吗？

### q108 — What do aq and rl mean on a RISC-V atomic memory operation?

Intent: Understand the memory-ordering effects of aq and rl on RISC-V atomic operations.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `atomic_add`

- [riscv-unprivileged, PDF p. 92](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=92) — 13.1. Specifying Ordering of Atomic Instructions

### q109 — For RISC-V CSRRS, how does using x0 differ from using another register that contains zero?

Intent: Determine how naming x0 versus using another zero-valued source register affects CSRRS writes.

Category: `comparison` · ISA: `riscv` · Comparison group: `csr_update`

- [riscv-unprivileged, PDF p. 69](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=69) — 6.1. CSR Instructions

### q117 — How do I return from a function using LoongArch JIRL?

Intent: Find the LoongArch JIRL operand combination used to return from a function.

Category: `procedure` · ISA: `loongarch` · Comparison group: `function_return`

- [loongarch-vol1, PDF p. 49](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=49) — 2.2.4.5. JIRL

### q118 — What does the DB suffix change for LoongArch atomic memory instructions?

Intent: Understand the additional ordering behavior of a LoongArch atomic instruction with the DB suffix.

Category: `comparison` · ISA: `loongarch` · Comparison group: `memory_barrier`

- [loongarch-vol1, PDF p. 62](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=62) — 2.2.7.1. Atomic memory access instructions

### q195 — Which LoongArch CPUCFG field indicates support for ARM binary translation?

Intent: Locate the CPUCFG capability indicator for ARM binary translation.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `cpu_identity`

- [loongarch-vol1, PDF p. 72](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=72) — 2.2.10.5. cpucfg, Table 3

### q196 — What is the RISC-V analogue of ARM's load-exclusive/store-exclusive approach for implementing a lock?

Intent: Find the RISC-V analogue of the ARM exclusive-load/store pattern for atomic updates.

Category: `comparison` · ISA: `riscv` · Comparison group: `conditional_store`

- [riscv-unprivileged, PDF p. 674](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=674) — A.5. Code Porting and Mapping Guidelines
- [riscv-unprivileged, PDF p. 93](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=93) — 13.2. Zalrsc Extension for Load-Reserved/Store-Conditional Instructions
- [riscv-unprivileged, PDF p. 94](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=94) — 13.2. Zalrsc Extension for Load-Reserved/Store-Conditional Instructions
- [riscv-unprivileged, PDF p. 96](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-unprivileged-20250508.pdf#page=96) — 13.3. Eventual Success of Store-Conditional Instructions

### q197 — I am porting ARM code to LoongArch. Which instructions order data accesses and synchronize instruction fetch?

Intent: Find LoongArch instructions for data-access ordering and instruction-fetch synchronization when porting code.

Category: `comparison` · ISA: `loongarch` · Comparison group: `memory_barrier`

- [loongarch-vol1, PDF p. 66](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=66) — 2.2.8.1. DBAR
- [loongarch-vol1, PDF p. 67](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=67) — 2.2.8.2. IBAR

## Privileges, exceptions and address translation

### q055 — user mode versus privileged execution levels

Intent: Find how user execution differs from privileged execution and what the privilege levels are for.

Category: `comparison` · ISA: `both` · Comparison group: `privilege_levels`

- [riscv-privileged, PDF p. 17](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=17) — 1.2. Privilege Levels
- [riscv-privileged, PDF p. 18](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=18) — 1.2. Privilege Levels
- [loongarch-vol1, PDF p. 108](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=108) — 4.1. Privilege Levels

### q056 — What are the roles of M, S and U modes in RISC-V?

Intent: Find the purpose and intended software roles of the RISC-V M, S, and U privilege modes.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `privilege_levels`

- [riscv-privileged, PDF p. 18](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=18) — 1.2. Privilege Levels

### q057 — What are the roles of PLV0 through PLV3 in LoongArch?

Intent: Find the access distinctions and intended use of LoongArch PLV0 through PLV3.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `privilege_levels`

- [loongarch-vol1, PDF p. 108](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=108) — 4.1. Privilege Levels

### q058 — privileged instruction executed from user mode

Intent: Find the execution rules and exception behavior for a user-mode attempt to use a privileged instruction.

Category: `conceptual` · ISA: `both` · Comparison group: `illegal_privileged`

- [riscv-privileged, PDF p. 17](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=17) — 1.2. Privilege Levels
- [riscv-privileged, PDF p. 19](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=19) — 2.1. CSR Address Mapping Conventions
- [loongarch-vol1, PDF p. 21](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=21) — 2.1.4. Exceptions and interrupts
- [loongarch-vol1, PDF p. 21](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=21) — 2.1.3.1. Privileged Resources Accessible by Applications

### q059 — What happens when RISC-V user code accesses a CSR without the required privilege?

Intent: Find the architectural outcome of a RISC-V user-mode CSR access that lacks the required privilege.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `illegal_privileged`

- [riscv-privileged, PDF p. 19](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=19) — 2.1. CSR Address Mapping Conventions

### q060 — What exception occurs when LoongArch user code executes an instruction that is not permitted at its privilege level?

Intent: Find the exception behavior when LoongArch user code executes an instruction disallowed at its current privilege level.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `illegal_privileged`

- [loongarch-vol1, PDF p. 21](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=21) — 2.1.4. Exceptions and interrupts
- [loongarch-vol1, PDF p. 21](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=21) — 2.1.3.1. Privileged Resources Accessible by Applications

### q061 — interrupt pending but handler not running

Intent: Find architectural conditions that can prevent a pending interrupt from entering its handler.

Category: `problem_description` · ISA: `both` · Comparison group: `interrupt_enable`

- [riscv-privileged, PDF p. 37](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=37) — 3.1.6.1. Privilege and Global Interrupt-Enable Stack in mstatus register
- [riscv-privileged, PDF p. 49](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=49) — 3.1.9. Machine Interrupt (mip and mie) Registers
- [loongarch-vol1, PDF p. 128](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=128) — 6.1.4. Process of Processor Responding to Interrupts
- [loongarch-vol1, PDF p. 142](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=142) — 7.4.5. Exception Configuration (ECFG)

### q062 — How do RISC-V pending bits, enable bits and global interrupt-enable state interact?

Intent: Find how RISC-V pending state, per-source enables, global enables, and current privilege determine interrupt delivery.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `interrupt_enable`

- [riscv-privileged, PDF p. 37](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=37) — 3.1.6.1. Privilege and Global Interrupt-Enable Stack in mstatus register
- [riscv-privileged, PDF p. 49](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=49) — 3.1.9. Machine Interrupt (mip and mie) Registers

### q063 — How do LoongArch CRMD.IE, ECFG and ESTAT control delivery of line-based interrupts?

Intent: Find how the named LoongArch registers combine to allow or block a pending line-based interrupt.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `interrupt_enable`

- [loongarch-vol1, PDF p. 128](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=128) — 6.1.4. Process of Processor Responding to Interrupts
- [loongarch-vol1, PDF p. 142](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=142) — 7.4.5. Exception Configuration (ECFG)

### q064 — How is the entry address of an exception or interrupt handler selected?

Intent: Find how processor configuration and trap type determine the address at which an exception or interrupt handler starts.

Category: `conceptual` · ISA: `both` · Comparison group: `handler_entry`

- [riscv-privileged, PDF p. 127](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=127) — 12.1.2. Supervisor Trap Vector Base Address (stvec) Register
- [loongarch-vol1, PDF p. 129](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=129) — 6.3.1. Exception Entry
- [loongarch-vol1, PDF p. 142](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=142) — 7.4.5. Exception Configuration (ECFG)

### q065 — How does RISC-V stvec choose direct or vectored trap entry?

Intent: Find how RISC-V stvec.MODE changes the target address for synchronous traps and interrupts.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `handler_entry`

- [riscv-privileged, PDF p. 127](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=127) — 12.1.2. Supervisor Trap Vector Base Address (stvec) Register

#### probe-ru-q065 (RU)

Как stvec в RISC-V выбирает прямой или векторный вход в обработчик?

#### probe-zh-q065 (ZH)

RISC-V 的 stvec 如何在直接模式和向量模式下确定陷入入口？

### q066 — How do LoongArch EENTRY and vector spacing select entry addresses for general exceptions and line-based interrupts?

Intent: Find how LoongArch EENTRY and vector spacing determine entry addresses for general exceptions and line-based interrupts.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `handler_entry`

- [loongarch-vol1, PDF p. 129](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=129) — 6.3.1. Exception Entry
- [loongarch-vol1, PDF p. 142](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=142) — 7.4.5. Exception Configuration (ECFG)

### q067 — saved program counter after an exception

Intent: Find which instruction address is saved when an exception occurs and how that saved address is used.

Category: `conceptual` · ISA: `both` · Comparison group: `saved_pc`

- [riscv-privileged, PDF p. 130](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=130) — 12.1.7. Supervisor Exception Program Counter (sepc) Register
- [loongarch-vol1, PDF p. 144](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=144) — 7.4.7. Exception Return Address (ERA)

### q068 — What value is saved in RISC-V sepc when a supervisor trap is taken?

Intent: Find the address recorded in RISC-V sepc when a trap enters supervisor mode.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `saved_pc`

- [riscv-privileged, PDF p. 130](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=130) — 12.1.7. Supervisor Exception Program Counter (sepc) Register

### q069 — What does LoongArch ERA record for a general exception?

Intent: Find which instruction address LoongArch records in ERA for a general exception.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `saved_pc`

- [loongarch-vol1, PDF p. 144](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=144) — 7.4.7. Exception Return Address (ERA)

### q070 — faulting virtual address register

Intent: Find which architectural state can identify the virtual address responsible for a memory-related fault.

Category: `conceptual` · ISA: `both` · Comparison group: `fault_address`

- [riscv-privileged, PDF p. 131](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=131) — 12.1.9. Supervisor Trap Value (stval) Register
- [loongarch-vol1, PDF p. 145](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=145) — 7.4.8. Bad Virtual Address (BADV)
- [loongarch-vol1, PDF p. 155](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=155) — 7.5.12. TLB Refill Exception Bad Virtual Address (TLBRBADV)

### q071 — When does RISC-V stval contain the faulting virtual address?

Intent: Find the conditions under which RISC-V stval provides a faulting virtual address and when that information may be absent.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `fault_address`

- [riscv-privileged, PDF p. 131](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=131) — 12.1.9. Supervisor Trap Value (stval) Register

### q072 — What address does LoongArch BADV record, and how does TLBRBADV differ?

Intent: Find what LoongArch BADV and TLBRBADV record and which fault contexts use each register.

Category: `comparison` · ISA: `loongarch` · Comparison group: `fault_address`

- [loongarch-vol1, PDF p. 145](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=145) — 7.4.8. Bad Virtual Address (BADV)
- [loongarch-vol1, PDF p. 155](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=155) — 7.5.12. TLB Refill Exception Bad Virtual Address (TLBRBADV)

#### probe-ru-q072 (RU)

Какой адрес хранит BADV в LoongArch и чем отличается TLBRBADV?

#### probe-zh-q072 (ZH)

LoongArch 的 BADV 记录什么地址？它与 TLBRBADV 有什么区别？

### q073 — restore privilege and interrupt state after an exception

Intent: Find how an exception return restores execution privilege and interrupt-enable state.

Category: `conceptual` · ISA: `both` · Comparison group: `exception_return`

- [riscv-privileged, PDF p. 37](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=37) — 3.1.6.1. Privilege and Global Interrupt-Enable Stack in mstatus register
- [riscv-privileged, PDF p. 65](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=65) — 3.3.2. Trap-Return Instructions
- [loongarch-vol1, PDF p. 130](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=130) — 6.3.3. General Hardware Exception Handling of General Exceptions

### q074 — What state does RISC-V SRET restore when returning from a supervisor trap?

Intent: Find the architectural state changes made by RISC-V SRET when returning from a supervisor trap.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `exception_return`

- [riscv-privileged, PDF p. 37](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=37) — 3.1.6.1. Privilege and Global Interrupt-Enable Stack in mstatus register
- [riscv-privileged, PDF p. 65](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=65) — 3.3.2. Trap-Return Instructions

### q075 — What state does LoongArch ERTN restore after an exception?

Intent: Find the architectural state restored by LoongArch ERTN for the exception being handled.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `exception_return`

- [loongarch-vol1, PDF p. 114](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=114) — 4.2.6.1. ERTN
- [loongarch-vol1, PDF p. 130](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=130) — 6.3.3. General Hardware Exception Handling of General Exceptions
- [loongarch-vol1, PDF p. 156](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=156) — 7.5.13. TLB Refill Exception Return Address (TLBRERA)

### q076 — address space identifiers and cached translations

Intent: Find how address-space identifiers distinguish translations and affect reuse or invalidation of cached mappings.

Category: `conceptual` · ISA: `both` · Comparison group: `asid`

- [riscv-privileged, PDF p. 134](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=134) — 12.1.11. Supervisor Address Translation and Protection (satp) Register
- [riscv-privileged, PDF p. 137](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=137) — 12.2.1. Supervisor Memory-Management Fence Instruction
- [loongarch-vol1, PDF p. 118](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=118) — 5.4.2. TLB Entry

### q077 — How is the ASID field used in RISC-V satp and SFENCE.VMA?

Intent: Find how the RISC-V satp ASID identifies an address space and how SFENCE.VMA uses an ASID when selecting translations to synchronize.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `asid`

- [riscv-privileged, PDF p. 134](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=134) — 12.1.11. Supervisor Address Translation and Protection (satp) Register
- [riscv-privileged, PDF p. 137](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=137) — 12.2.1. Supervisor Memory-Management Fence Instruction

### q078 — How is the LoongArch ASID register used in address translation and TLB matching?

Intent: Find how LoongArch uses the ASID register when looking up virtual-address translations.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `asid`

- [loongarch-vol1, PDF p. 118](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=118) — 5.4.2. TLB Entry
- [loongarch-vol1, PDF p. 152](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=152) — 7.5.4. Address Space Identifier (ASID)

### q079 — page table read write execute and user permissions

Intent: Find how page-mapping permissions restrict reads, writes, instruction fetches, and access from less-privileged software.

Category: `conceptual` · ISA: `both` · Comparison group: `page_permissions`

- [riscv-privileged, PDF p. 141](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=141) — 12.3.1. Addressing and Memory Protection
- [loongarch-vol1, PDF p. 119](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=119) — 5.4.2. TLB Entry
- [loongarch-vol1, PDF p. 120](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=120) — 5.4.3.1. TLB-related Exceptions

### q080 — How do the R, W, X and U bits control access to RISC-V pages?

Intent: Find the meaning and interaction of RISC-V page-entry R, W, X, and U permissions.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `page_permissions`

- [riscv-privileged, PDF p. 141](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=141) — 12.3.1. Addressing and Memory Protection

### q081 — How do LoongArch LA64 page-entry privilege fields and NR/NX bits restrict access?

Intent: Find how LoongArch LA64 page entries restrict access by privilege level and by load or instruction-fetch operation.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `page_permissions`

- [loongarch-vol1, PDF p. 119](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=119) — 5.4.2. TLB Entry

### q082 — large page mappings and page-size selection

Intent: Find how architectures represent larger page mappings and select their page size.

Category: `conceptual` · ISA: `both` · Comparison group: `large_pages`

- [riscv-privileged, PDF p. 147](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=147) — 12.4.1. Addressing and Memory Protection
- [loongarch-vol1, PDF p. 119](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=119) — 5.4.2. TLB Entry
- [loongarch-vol1, PDF p. 118](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=118) — 5.4.2. TLB Entry
- [loongarch-vol1, PDF p. 125](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=125) — 5.4.5. Multi-level Page Table Structure Supported by page walking

### q083 — What makes a RISC-V Sv39 leaf a superpage, and what alignment is required?

Intent: Find how a higher-level Sv39 leaf maps a larger page and which address-alignment conditions apply.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `large_pages`

- [riscv-privileged, PDF p. 147](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=147) — 12.4.1. Addressing and Memory Protection

### q084 — How are page sizes represented in LoongArch TLB entries and page-table walking?

Intent: Find how LoongArch encodes page sizes in TLB state and recognizes huge-page mappings while walking page tables.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `large_pages`

- [loongarch-vol1, PDF p. 119](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=119) — 5.4.2. TLB Entry
- [loongarch-vol1, PDF p. 118](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=118) — 5.4.2. TLB Entry
- [loongarch-vol1, PDF p. 125](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=125) — 5.4.5. Multi-level Page Table Structure Supported by page walking

### q085 — invalidate stale translations after changing a page table

Intent: Find how software makes cached address translations consistent after modifying a page table.

Category: `procedure` · ISA: `both` · Comparison group: `translation_sync`

- [riscv-privileged, PDF p. 137](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=137) — 12.2.1. Supervisor Memory-Management Fence Instruction
- [loongarch-vol1, PDF p. 112](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=112) — 4.2.4.7. INVTLB

### q086 — What does RISC-V SFENCE.VMA synchronize after a page-table update?

Intent: Find the ordering and invalidation guarantees of RISC-V SFENCE.VMA after page-table modifications.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `translation_sync`

- [riscv-privileged, PDF p. 137](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=137) — 12.2.1. Supervisor Memory-Management Fence Instruction

### q087 — How does LoongArch INVTLB invalidate translations after mapping changes?

Intent: Find how to select the translations invalidated by LoongArch INVTLB after changing mappings.

Category: `procedure` · ISA: `loongarch` · Comparison group: `translation_sync`

- [loongarch-vol1, PDF p. 112](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=112) — 4.2.4.7. INVTLB
- [loongarch-vol1, PDF p. 113](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=113) — 4.2.4.7. INVTLB

### q088 — configure and acknowledge a timer interrupt

Intent: Find how to arrange a timer interrupt and acknowledge or clear it after delivery.

Category: `procedure` · ISA: `both` · Comparison group: `timer_interrupt`

- [riscv-privileged, PDF p. 63](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=63) — 3.2.1. Machine Timer (mtime and mtimecmp) Registers
- [loongarch-vol1, PDF p. 161](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=161) — 7.6.2. Timer Configuration (TCFG)
- [loongarch-vol1, PDF p. 161](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=161) — 7.6.5. Timer Interrupt Clearing (TICLR)

### q089 — How do RISC-V mtime and mtimecmp determine the machine timer interrupt?

Intent: Find how RISC-V mtime and mtimecmp make a machine-timer interrupt pending and clear that pending condition.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `timer_interrupt`

- [riscv-privileged, PDF p. 63](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=63) — 3.2.1. Machine Timer (mtime and mtimecmp) Registers

### q090 — How do LoongArch TCFG, TVAL and TICLR control timer interrupts?

Intent: Find the respective roles of LoongArch TCFG, TVAL, and TICLR when using a timer interrupt.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `timer_interrupt`

- [loongarch-vol1, PDF p. 161](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=161) — 7.6.2. Timer Configuration (TCFG)
- [loongarch-vol1, PDF p. 161](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=161) — 7.6.3. Timer Value (TVAL)
- [loongarch-vol1, PDF p. 161](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=161) — 7.6.5. Timer Interrupt Clearing (TICLR)

### q094 — If several exceptions are possible for one operation, how is the reported exception selected?

Intent: Find which exception is reported when one operation meets conditions for more than one exception.

Category: `conceptual` · ISA: `both` · Comparison group: `exception_priority`

- [riscv-privileged, PDF p. 57](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=57) — 3.1.15. Machine Cause (mcause) Register; Table 15
- [loongarch-vol1, PDF p. 129](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=129) — 6.3.2. Exception Priority

### q096 — How are cacheable memory and device-style accesses distinguished by the architecture?

Intent: Find the architectural memory attributes that distinguish cacheable memory accesses from accesses intended for devices.

Category: `conceptual` · ISA: `both` · Comparison group: `memory_types`

- [riscv-privileged, PDF p. 69](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=69) — 3.6.1. Main Memory versus I/O Regions
- [riscv-privileged, PDF p. 72](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=72) — 3.6.6. Coherence and Cacheability PMAs
- [riscv-privileged, PDF p. 152](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=152) — 14. "Svpbmt" Extension for Page-Based Memory Types, Version 1.0
- [loongarch-vol1, PDF p. 22](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=22) — 2.1.7. Memory Access Types

### q098 — What does the processor's wait-for-interrupt or idle instruction guarantee about resuming execution?

Intent: Find what software may assume about suspension and resumption around a wait-for-interrupt or idle instruction.

Category: `conceptual` · ISA: `both` · Comparison group: `idle_wait`

- [riscv-privileged, PDF p. 65](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=65) — 3.3.3. Wait for Interrupt
- [riscv-privileged, PDF p. 66](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=66) — 3.3.3. Wait for Interrupt
- [loongarch-vol1, PDF p. 115](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=115) — 4.2.6.3. IDLE

### q099 — How is a virtual address translated into a physical address through multiple page-table levels?

Intent: Find how the processor or software walks successive page-table levels to translate a virtual address.

Category: `conceptual` · ISA: `both` · Comparison group: `address_translation`

- [riscv-privileged, PDF p. 146](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=146) — 12.4.1. Addressing and Memory Protection
- [riscv-privileged, PDF p. 144](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=144) — 12.3.2. Virtual Address Translation Process
- [loongarch-vol1, PDF p. 124](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=124) — 5.4.5. Multi-level Page Table Structure Supported by page walking
- [loongarch-vol1, PDF p. 125](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=125) — 5.4.5. Multi-level Page Table Structure Supported by page walking

### q100 — Is a missing TLB entry necessarily the same as an invalid virtual-memory mapping?

Intent: Find the distinction between a missing cached translation and a virtual address that has no valid mapping.

Category: `comparison` · ISA: `both` · Comparison group: `tlb_miss`

- [riscv-privileged, PDF p. 145](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=145) — 12.3.2. Virtual Address Translation Process
- [riscv-privileged, PDF p. 144](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=144) — 12.3.2. Virtual Address Translation Process
- [loongarch-vol1, PDF p. 120](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=120) — 5.4.3.1. TLB-related Exceptions
- [hailperin-osm, PDF p. 250](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=250) — 6.3. Mechanisms for Virtual Memory

### q101 — RISC-V satp write with an unsupported MODE value

Intent: Find what happens to the RISC-V satp register when software writes an address-translation MODE that the implementation does not support.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `satp_unsupported_mode`

- [riscv-privileged, PDF p. 135](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=135) — 12.1.11. Supervisor Address Translation and Protection (satp) Register

### q102 — RISC-V stvec BASE alignment requirements in direct and vectored modes

Intent: Find the alignment requirements for a RISC-V stvec base address in direct and vectored modes.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `stvec_alignment`

- [riscv-privileged, PDF p. 127](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=127) — 12.1.2. Supervisor Trap Vector Base Address (stvec) Register

### q103 — How can RISC-V software determine which exceptions medeleg can delegate to S-mode?

Intent: Find how RISC-V software discovers which synchronous exceptions the implementation allows medeleg to delegate to supervisor mode.

Category: `procedure` · ISA: `riscv` · Comparison group: `delegation`

- [riscv-privileged, PDF p. 47](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=47) — 3.1.8. Machine Trap Delegation (medeleg and mideleg) Registers
- [riscv-privileged, PDF p. 48](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=48) — 3.1.8. Machine Trap Delegation (medeleg and mideleg) Registers

### q104 — How does RISC-V SUM affect supervisor access to user pages?

Intent: Find how the RISC-V SUM setting affects supervisor accesses to user-accessible pages.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `page_permissions`

- [riscv-privileged, PDF p. 125](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=125) — 12.1.1.2. Memory Privilege in sstatus Register

### q105 — How does RISC-V MXR affect reads from execute-only pages?

Intent: Find how the RISC-V MXR setting affects loads from pages that permit execution but not ordinary reads.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `page_permissions`

- [riscv-privileged, PDF p. 124](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=124) — 12.1.1.2. Memory Privilege in sstatus Register

### q106 — RISC-V: I changed satp but still see an old mapping. What synchronization is required?

Intent: Find what synchronization is needed when changing RISC-V satp leaves translations from the previous mapping visible.

Category: `problem_description` · ISA: `riscv` · Comparison group: `translation_sync`

- [riscv-privileged, PDF p. 136](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=136) — 12.1.11. Supervisor Address Translation and Protection (satp) Register
- [riscv-privileged, PDF p. 137](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=137) — 12.2.1. Supervisor Memory-Management Fence Instruction

### q110 — How do RISC-V PMP entries decide whether a physical memory access is permitted?

Intent: Find how RISC-V PMP matching, entry priority, access size, and permission bits determine whether a physical-memory access is allowed.

Category: `conceptual` · ISA: `riscv` · Comparison group: `physical_protection`

- [riscv-privileged, PDF p. 77](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=77) — 3.7.1.3. Priority and Matching Logic

### q111 — LoongArch EENTRY versus TLBRENTRY

Intent: Find the difference between LoongArch EENTRY and TLBRENTRY, including which exception types and address interpretations each uses.

Category: `comparison` · ISA: `loongarch` · Comparison group: `handler_entry`

- [loongarch-vol1, PDF p. 145](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=145) — 7.4.10. Exception Entry Base Address (EENTRY)
- [loongarch-vol1, PDF p. 155](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=155) — 7.5.11. TLB Refill Exception Entry Base Address (TLBRENTRY)

### q112 — LoongArch PRMD PPLV PIE saved exception state

Intent: Find what LoongArch PRMD.PPLV and PRMD.PIE save at exception entry and how the saved state is restored.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `exception_return`

- [loongarch-vol1, PDF p. 130](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=130) — 6.3.3. General Hardware Exception Handling of General Exceptions

### q113 — When does LoongArch use PGDL rather than PGDH?

Intent: Find when LoongArch selects the lower-half rather than higher-half page-directory base.

Category: `comparison` · ISA: `loongarch` · Comparison group: `address_translation`

- [loongarch-vol1, PDF p. 153](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=153) — 7.5.7. Page Global Directory Base Address (PGD)

### q114 — LoongArch CRMD DA PG address translation modes

Intent: Find which LoongArch address-translation modes are selected by the CRMD.DA and CRMD.PG settings.

Category: `comparison` · ISA: `loongarch` · Comparison group: `address_translation`

- [loongarch-vol1, PDF p. 116](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=116) — 5.2. Virtual Address Space and Address Translation Mode

### q115 — LoongArch timer interrupt remains pending: how do I clear it?

Intent: Find how to clear a pending LoongArch timer interrupt.

Category: `problem_description` · ISA: `loongarch` · Comparison group: `timer_interrupt`

- [loongarch-vol1, PDF p. 161](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=161) — 7.6.5. Timer Interrupt Clearing (TICLR)

### q116 — Why does LoongArch have TLBELO0 and TLBELO1 for a TLB entry?

Intent: Find why one LoongArch TLB entry uses the two TLBELO registers and how the two registers correspond to the mapping.

Category: `comparison` · ISA: `loongarch` · Comparison group: `tlb_entry_pair`

- [loongarch-vol1, PDF p. 150](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=150) — 7.5.3. TLB Entry Low-order Bits (TLBELO0, TLBELO1)

### q119 — What is the purpose of LoongArch TLBRERA.IsTLBR?

Intent: Find the purpose of LoongArch TLBRERA.IsTLBR and how it affects exception-state handling.

Category: `exact_lookup` · ISA: `loongarch` · Comparison group: `tlb_miss`

- [loongarch-vol1, PDF p. 156](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=156) — 7.5.13. TLB Refill Exception Return Address (TLBRERA)

### q120 — How does LoongArch LA64 RPLV change the privilege check for a TLB entry?

Intent: Find how RPLV changes which privilege levels may access a LoongArch LA64 TLB mapping.

Category: `comparison` · ISA: `loongarch` · Comparison group: `page_permissions`

- [loongarch-vol1, PDF p. 119](https://raw.githubusercontent.com/loongson/LoongArch-Documentation/298e9977e7d6368cbd1a095434512072601cccc7/LoongArch-Vol1-EN.pdf#page=119) — 5.4.2. TLB Entry

#### probe-ru-q120 (RU)

Как RPLV в LoongArch LA64 меняет проверку привилегий при использовании записи TLB?

#### probe-zh-q120 (ZH)

LoongArch LA64 的 RPLV 如何改变 TLB 项的特权级检查？

### q199 — In RISC-V, what happens to floating-point register contents when software sets FS to Initial?

Intent: Find the effect of setting RISC-V FS to Initial on floating-point register contents.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `fp_context_state`

- [riscv-privileged, PDF p. 43](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=43) — 3.1.6.7. Extension Context Status in mstatus Register
- [riscv-privileged, PDF p. 44](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=44) — 3.1.6.7. Extension Context Status in mstatus Register

### q200 — Under what conditions can non-virtualized RISC-V supervisor-mode software access stimecmp?

Intent: Find the conditions governing access to stimecmp from non-virtualized RISC-V supervisor mode.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `supervisor_timer_access`

- [riscv-privileged, PDF p. 160](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=160) — 19.1.1. Supervisor Timer (stimecmp) Register
- [riscv-privileged, PDF p. 161](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=161) — 19.1.4. Machine Counter-Enable (mcounteren) Register
- [riscv-privileged, PDF p. 162](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=162) — 19.3. Environment Config (menvcfg and henvcfg) Support

## Operating systems, xv6, files, networking and security

### q121 — Why does fork return different values in the parent and the child?

Intent: Understand the purpose and meaning of the two return values from fork.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `process_creation`

- [hailperin-osm, PDF p. 296](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=296) — 7.2 POSIX Process Management API

### q122 — Why does a shell call exec after fork instead of replacing itself directly?

Intent: Understand why a shell creates a child before executing a command.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `process_creation`

- [xv6-riscv, PDF p. 12](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=12) — 1.1 Processes and memory

### q123 — What resources or state remain when a child exits before its parent calls wait?

Intent: Find what remains of an exited child until the parent collects its termination status.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `process_wait`

- [xv6-riscv, PDF p. 85](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=85) — 9.4 Code: Wait, exit, and kill

### q124 — After fork, can a seek through one process's file descriptor affect the other process?

Intent: Determine whether parent and child file-position changes can affect one another after fork.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `file_descriptors`

- [xv6-riscv, PDF p. 15](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=15) — 1.2 I/O and File descriptors
- [hailperin-osm, PDF p. 369](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=369) — 8.3.4 Sequential Reading and Writing

### q125 — pipe read never reaches EOF after a child exits

Intent: Diagnose a pipe reader that does not observe end-of-file after a child terminates.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `pipe_eof`

- [xv6-riscv, PDF p. 16](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=16) — 1.3 Pipes

#### probe-ru-q125 (RU)

чтение из pipe не доходит до EOF после завершения дочернего процесса

#### probe-zh-q125 (ZH)

子进程退出后，读管道一直读不到 EOF

### q126 — Why can frequent context switches improve response time while reducing throughput?

Intent: Understand the scheduling tradeoff between aggregate work completed and interactive responsiveness.

Category: `comparison` · ISA: `unspecified` · Comparison group: `scheduling_metrics`

- [hailperin-osm, PDF p. 74](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=74) — 3.3.2 Response Time

### q127 — Why does a decay-usage scheduler favor a thread that wakes after waiting over one that has recently used the CPU?

Intent: Understand why recent CPU usage changes the priority of a thread that wakes after waiting.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `decay_usage`

- [hailperin-osm, PDF p. 88](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=88) — 3.5.2 Decay Usage Scheduling

### q128 — How does lottery scheduling turn weighted random choices into CPU shares?

Intent: Understand how random dispatch can implement requested CPU allocations and over what timescale.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `lottery`

- [hailperin-osm, PDF p. 91](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=91) — 3.6 Proportional-Share Scheduling
- [hailperin-osm, PDF p. 92](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=92) — 3.6 Proportional-Share Scheduling

### q129 — Why can moving a thread between CPUs hurt cache performance?

Intent: Understand the performance cost of moving a thread between processors.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `multicore_scheduling`

- [hailperin-osm, PDF p. 72](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=72) — 3.3.1 Throughput

### q130 — In xv6, what does swtch save, and how does that differ from the user trap frame?

Intent: Compare the saved execution state of an xv6 kernel context switch with a trap from user space.

Category: `comparison` · ISA: `riscv` · Comparison group: `context_switch_state`

- [xv6-riscv, PDF p. 77](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=77) — 8.3 Code: Context switching
- [xv6-riscv, PDF p. 46](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=46) — 4.2 Traps from user space

### q131 — internal versus external fragmentation

Intent: Distinguish internal and external fragmentation.

Category: `comparison` · ISA: `unspecified` · Comparison group: `fragmentation`

- [hailperin-osm, PDF p. 371](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=371) — 8.4.1 Fragmentation
- [hailperin-osm, PDF p. 372](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=372) — 8.4.1 Fragmentation

### q132 — Why can a sparse virtual address space occupy much less physical memory than its address range suggests?

Intent: Understand the relation between the span of a sparse virtual address space and its physical-memory use.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `sparse_address_space`

- [hailperin-osm, PDF p. 242](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=242) — 6.2.4 Sparse Address Spaces

### q133 — How does a multi-level page table save space for a sparse address space?

Intent: Understand the space advantage of a multilevel page table when large virtual ranges are unused.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `page_table_size`

- [xv6-riscv, PDF p. 32](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=32) — 3.1 Paging hardware

### q134 — array access gets much slower when consecutive accesses are one page apart

Intent: Investigate possible reasons for slow array access when successive accesses are one memory page apart.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `tlb_locality`

- [hailperin-osm, PDF p. 250](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=250) — 6.3.1 Software/Hardware Interface

### q135 — Why can FIFO page replacement perform worse after more page frames are added?

Intent: Understand how increasing page-frame capacity can worsen FIFO page replacement.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `replacement`

- [hailperin-osm, PDF p. 279](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=279) — 6.4.3 Replacement Policy
- [hailperin-osm, PDF p. 280](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=280) — 6.4.3 Replacement Policy

### q136 — constant paging but little useful work gets done

Intent: Investigate why a system does little useful work while continually paging.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `working_set`

- [hailperin-osm, PDF p. 278](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=278) — 6.4.3 Replacement Policy

### q137 — How can copy-on-write fork share pages without letting parent and child overwrite each other's memory?

Intent: Understand how parent and child can share physical memory while preserving fork isolation.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `copy_on_write`

- [xv6-riscv, PDF p. 53](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=53) — 5.3 Real world: Copy-On-Write (COW) fork

### q138 — Why can xv6 sbrk with SBRK_LAZY succeed before physical memory is allocated, and what happens on first access?

Intent: Understand when xv6 reserves address space without allocating physical pages and what first access does.

Category: `conceptual` · ISA: `riscv` · Comparison group: `lazy_allocation`

- [xv6-riscv, PDF p. 51](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=51) — 5.1 Lazy allocation
- [xv6-riscv, PDF p. 52](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=52) — 5.2 Code

### q139 — How can I distinguish a page permission fault from a missing mapping?

Intent: Determine how to investigate whether a page fault is caused by a missing mapping or insufficient access permissions.

Category: `comparison` · ISA: `unspecified` · Comparison group: `page_permissions`

- [xv6-riscv, PDF p. 51](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=51) — 5 Page faults
- [xv6-riscv, PDF p. 33](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=33) — 3.1 Paging hardware

### q140 — Why does a TLB miss not necessarily require reading a page from disk?

Intent: Distinguish the work required to resolve a TLB miss from paging data in from disk.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `tlb_miss`

- [hailperin-osm, PDF p. 250](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=250) — 6.3.1 Software/Hardware Interface
- [xv6-riscv, PDF p. 32](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=32) — 3.1 Paging hardware

### q141 — spin lock versus sleeping lock under contention

Intent: Compare spinning and sleeping while waiting to acquire a contended lock.

Category: `comparison` · ISA: `unspecified` · Comparison group: `lock_contention`

- [xv6-riscv, PDF p. 72](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=72) — 7.7 Sleep locks

### q142 — Why should a condition-variable wait usually be inside a while loop?

Intent: Find why condition-variable waits are usually placed inside a while loop.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `condition_variables`

- [hailperin-osm, PDF p. 140](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=140) — 4.5 Condition Variables

#### probe-ru-q142 (RU)

Почему ожидание условной переменной обычно нужно помещать в цикл while?

#### probe-zh-q142 (ZH)

为什么条件变量的等待通常应该放在 while 循环里？

### q143 — condition changed and wakeup ran, but the waiting thread sleeps forever

Intent: Diagnose a thread that sleeps indefinitely despite a condition change and an earlier wakeup.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `sleep_wakeup_race`

- [xv6-riscv, PDF p. 82](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=82) — 9.1 Overview

### q144 — How can a semaphore initialized to zero coordinate completion between two threads?

Intent: Find how to use a zero-initialized semaphore to order completion and continuation in two threads.

Category: `procedure` · ISA: `unspecified` · Comparison group: `semaphore`

- [hailperin-osm, PDF p. 143](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=143) — 4.6 Semaphores

### q145 — Which conditions are necessary for deadlock, and how can lock ordering break one of them?

Intent: Find the necessary conditions for deadlock and how lock ordering relates to them.

Category: `procedure` · ISA: `unspecified` · Comparison group: `deadlock`

- [hailperin-osm, PDF p. 147](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=147) — 4.7.1 The Deadlock Problem
- [hailperin-osm, PDF p. 149](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=149) — 4.7.2 Deadlock Prevention Through Resource Ordering

### q146 — low priority thread blocks a high priority thread through a lock

Intent: Find why a low-priority thread can delay a high-priority thread through a lock.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `priority_inversion`

- [hailperin-osm, PDF p. 156](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=156) — 4.8.1 Priority Inversion

### q147 — two threads increment a shared counter, but the final total is too small

Intent: Investigate why concurrent increments by two threads can leave a shared counter below the expected total.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `atomic_add`

- [hailperin-osm, PDF p. 116](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=116) — 4.2 Races and the Need for Mutual Exclusion
- [xv6-riscv, PDF p. 65](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=65) — 7.1 Races

### q148 — Why can an atomic-exchange spinlock generate heavy cache-coherence traffic under contention?

Intent: Understand the cache-coherence cost of repeatedly contending for an atomic-exchange spinlock.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `lock_contention`

- [hailperin-osm, PDF p. 127](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=127) — 4.3.3 Underlying Mechanisms for Mutexes

### q149 — Why does disabling interrupts on one CPU not protect shared data from other CPUs?

Intent: Understand the limits of disabling local interrupts as synchronization on a multiprocessor.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `interrupt_locks`

- [xv6-riscv, PDF p. 71](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=71) — 7.5 Locks and interrupts
- [xv6-riscv, PDF p. 65](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=65) — 7.1 Races

### q150 — Can a lock built with atomic instructions still need memory barriers?

Intent: Determine whether atomic lock acquisition alone is enough to order accesses within a critical section.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `lock_memory_order`

- [xv6-riscv, PDF p. 67](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=67) — 7.2 Code: Locks
- [xv6-riscv, PDF p. 71](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=71) — 7.6 Instruction and memory ordering

### q151 — Why is the xv6 trampoline mapped at the same virtual address in user and kernel page tables?

Intent: Understand why xv6 uses the same trampoline virtual address across page-table changes.

Category: `conceptual` · ISA: `riscv` · Comparison group: `trampoline`

- [xv6-riscv, PDF p. 46](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=46) — 4.2 Traps from user space

### q152 — xv6 sscratch user trap entry

Intent: Find the role of sscratch at the start of xv6 user-trap handling.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `trampoline`

- [xv6-riscv, PDF p. 46](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=46) — 4.2 Traps from user space

### q153 — How does xv6 safely copy a system-call argument string out of user memory?

Intent: Find how xv6 safely retrieves a string supplied as a user-space system-call argument.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `user_memory_access`

- [xv6-riscv, PDF p. 48](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=48) — 4.4 Code: System call arguments

### q154 — How do xv6's walk loop and the RISC-V Sv39 virtual-address fields correspond?

Intent: Relate the levels and address bits used by xv6 walk to the Sv39 translation format.

Category: `comparison` · ISA: `riscv` · Comparison group: `page_walk_code`

- [xv6-riscv, PDF p. 36](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=36) — 3.3 Code: creating an address space
- [xv6-riscv, PDF p. 32](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=32) — 3.1 Paging hardware
- [riscv-privileged, PDF p. 146](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=146) — 12.4.1 Addressing and Memory Protection

### q155 — How does an unmapped guard page help detect a kernel stack overflow in xv6?

Intent: Understand how xv6 uses a guard page to detect kernel-stack overflow.

Category: `conceptual` · ISA: `riscv` · Comparison group: `guard_page`

- [xv6-riscv, PDF p. 35](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=35) — 3.2 Kernel address space

### q156 — Why does accept return a new socket instead of turning the listening socket into the connection?

Intent: Understand the relationship between a listening TCP socket and the socket returned by accept.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `socket_accept`

- [hailperin-osm, PDF p. 434](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=434) — 9.3.1 Socket APIs

### q157 — Why does xv6 transfer a process lock across swtch between a process and the scheduler?

Intent: Understand why the xv6 process lock is handed across a context switch to the scheduler.

Category: `conceptual` · ISA: `riscv` · Comparison group: `context_switch_state`

- [xv6-riscv, PDF p. 78](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=78) — 8.4 Code: Scheduling

### q158 — How does xv6 sleep avoid losing a wakeup while releasing a caller's lock?

Intent: Find how xv6 sleep releases the condition lock without introducing a lost wakeup.

Category: `conceptual` · ISA: `riscv` · Comparison group: `sleep_wakeup_race`

- [xv6-riscv, PDF p. 83](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=83) — 9.2 Code: Sleep and wakeup

### q159 — In xv6 user trap entry, what does changing satp do, and why is SFENCE.VMA also used?

Intent: Understand the page-table switch and translation synchronization in xv6 user-trap entry.

Category: `conceptual` · ISA: `riscv` · Comparison group: `trampoline`

- [xv6-riscv, PDF p. 46](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=46) — 4.2 Traps from user space
- [xv6-riscv, PDF p. 37](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=37) — 3.3 Code: creating an address space
- [riscv-privileged, PDF p. 136](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=136) — 12.1.11 Supervisor Address Translation and Protection (satp) Register
- [riscv-privileged, PDF p. 137](https://github.com/riscv/riscv-isa-manual/releases/download/20250508/riscv-privileged-20250508.pdf#page=137) — 12.2.1 Supervisor Memory-Management Fence Instruction

### q160 — Why does xv6 disable interrupts while holding spinlocks?

Intent: Understand why xv6 disables interrupts while holding a spinlock.

Category: `conceptual` · ISA: `riscv` · Comparison group: `interrupt_locks`

- [xv6-riscv, PDF p. 70](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=70) — 7.5 Locks and interrupts
- [xv6-riscv, PDF p. 71](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=71) — 7.5 Locks and interrupts

### q161 — How do an inode, a directory entry and a file descriptor refer to different layers of a file?

Intent: Compare the roles of an inode, a directory entry and a file descriptor.

Category: `comparison` · ISA: `unspecified` · Comparison group: `filesystem_names`

- [xv6-riscv, PDF p. 99](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=99) — 10.11 Code: directory layer
- [xv6-riscv, PDF p. 18](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=18) — 1.4 File system
- [xv6-riscv, PDF p. 101](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=101) — 10.13 File descriptor layer

### q162 — file still readable after unlink

Intent: Investigate why a file can still be read after unlink.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `unlink_open`

- [xv6-riscv, PDF p. 18](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=18) — 1.4 File system

#### probe-ru-q162 (RU)

файл всё ещё читается после unlink

#### probe-zh-q162 (ZH)

unlink 后文件仍然可以读取

### q163 — How do a hard link and a symbolic link differ when the original pathname is removed?

Intent: Compare hard-link and symbolic-link behavior when the original pathname is removed.

Category: `comparison` · ISA: `unspecified` · Comparison group: `hard_links`

- [hailperin-osm, PDF p. 395](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=395) — 8.6.3 File Linking
- [hailperin-osm, PDF p. 396](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=396) — 8.6.3 File Linking

### q164 — What does fsync add beyond a successful write system call?

Intent: Understand what persistence guarantee an application seeks by calling fsync after writing.

Category: `comparison` · ISA: `unspecified` · Comparison group: `fsync`

- [hailperin-osm, PDF p. 399](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=399) — 8.7 Metadata Integrity

### q165 — How does a write-ahead log make a filesystem operation recoverable after a crash?

Intent: Understand how logging supports atomic recovery of filesystem updates after a crash.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `filesystem_log`

- [xv6-riscv, PDF p. 93](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=93) — 10.4 Logging layer

### q166 — Why can xv6 begin_op wait before allowing a filesystem operation to proceed?

Intent: Find the conditions under which xv6 delays entry into a filesystem transaction.

Category: `conceptual` · ISA: `riscv` · Comparison group: `filesystem_log`

- [xv6-riscv, PDF p. 94](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=94) — 10.6 Code: logging

### q167 — Why does xv6 keep only one cached buffer for a given disk block?

Intent: Understand the correctness reason for keeping one cached buffer per disk block in xv6.

Category: `conceptual` · ISA: `riscv` · Comparison group: `buffer_cache`

- [xv6-riscv, PDF p. 92](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=92) — 10.3 Code: Buffer cache

### q168 — Why are xv6 inode reference counts and inode locks separate mechanisms?

Intent: Understand how inode reference ownership differs from exclusive inode locking in xv6.

Category: `comparison` · ISA: `riscv` · Comparison group: `inode_lifetime`

- [xv6-riscv, PDF p. 96](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=96) — 10.8 Inode layer

### q169 — How does xv6 resolve an absolute pathname differently from a relative pathname?

Intent: Find where xv6 starts pathname resolution for absolute and relative paths.

Category: `comparison` · ISA: `riscv` · Comparison group: `pathname_resolution`

- [xv6-riscv, PDF p. 100](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=100) — 10.12 Code: Path names

### q170 — After a crash, how does xv6 decide whether to replay or discard its on-disk log?

Intent: Find the concrete on-disk condition xv6 uses to decide whether recovery replays a log transaction.

Category: `exact_lookup` · ISA: `riscv` · Comparison group: `filesystem_log`

- [xv6-riscv, PDF p. 93](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=93) — 10.5 Log design
- [xv6-riscv, PDF p. 93](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=93) — 10.4 Logging layer

### q171 — Why can two reads of the same size from a rotating hard disk take very different amounts of time?

Intent: Understand why equal-sized reads from a rotating hard disk can have very different latency.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `disk_access_cost`

- [hailperin-osm, PDF p. 358](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=358) — 8.2 Disk Storage Technology

### q172 — What does pread do differently from lseek followed by read?

Intent: Compare pread with an lseek call followed by read.

Category: `comparison` · ISA: `unspecified` · Comparison group: `pread_offset`

- [hailperin-osm, PDF p. 368](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=368) — 8.3.3 Reading and Writing Files at Specified Positions
- [hailperin-osm, PDF p. 370](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=370) — 8.3.4 Sequential Reading and Writing

### q173 — large file needs more data-block addresses than fit in its inode

Intent: Find how a file can be represented when its data-block addresses do not all fit in the inode.

Category: `problem_description` · ISA: `unspecified` · Comparison group: `indirect_file_blocks`

- [xv6-riscv, PDF p. 99](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=99) — 10.10 Code: Inode content

#### probe-ru-q173 (RU)

для большого файла нужно больше адресов блоков данных, чем помещается в inode

#### probe-zh-q173 (ZH)

大文件的数据块地址太多，inode 放不下

### q174 — Why is a disk block marked both allocated and free more dangerous than a lost block after a crash?

Intent: Compare the consequences of a lost block with a block that is both in use and marked free after a crash.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `metadata_integrity`

- [hailperin-osm, PDF p. 400](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=400) — 8.7 Metadata Integrity
- [xv6-riscv, PDF p. 92](https://pdos.csail.mit.edu/6.1810/2025/xv6/book-riscv-rev5.pdf#page=92) — 10.4 Logging layer

### q175 — A program opens a remote file through its normal pathname; where does the kernel dispatch the operation to the network filesystem?

Intent: Find how an ordinary file operation is routed to a network filesystem implementation.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `vfs_dispatch`

- [hailperin-osm, PDF p. 430](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=430) — 9.2.3 Distributed File Systems: An Application Viewed Through Operating Systems

### q176 — Why must a crash-recovery undo operation still work correctly if it is executed more than once?

Intent: Find why repeating a crash-recovery undo operation must remain safe.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `undo_idempotence`

- [hailperin-osm, PDF p. 209](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=209) — 5.4 Transaction Durability: Write-Ahead Logging

### q177 — How do access control lists and capabilities represent authority differently?

Intent: Compare how capabilities and access control lists associate authority with subjects and objects.

Category: `comparison` · ISA: `unspecified` · Comparison group: `access_control`

- [hailperin-osm, PDF p. 319](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=319) — 7.4.2 Capabilities
- [hailperin-osm, PDF p. 323](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=323) — 7.4.3 Access Control Lists and Credentials

### q178 — Why are authentication and authorization separate decisions?

Intent: Understand why authentication and authorization are treated as separate security decisions.

Category: `comparison` · ISA: `unspecified` · Comparison group: `authentication`

- [hailperin-osm, PDF p. 491](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=491) — 11.2 Security Objectives and Principles
- [hailperin-osm, PDF p. 323](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=323) — 7.4.3 Access Control Lists and Credentials

### q179 — In trap-and-emulate virtualization, why must the VMM intercept privileged guest instructions?

Intent: Understand why trap-and-emulate virtualization requires the monitor to take control of privileged guest operations.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `virtualization`

- [hailperin-osm, PDF p. 335](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=335) — 7.5.2 Protection of Entire Simulated Machines

### q180 — How does a monitor combine mutual exclusion with waiting for a condition?

Intent: Understand how monitor-style synchronization combines exclusive access with condition waiting.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `monitor`

- [hailperin-osm, PDF p. 124](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=124) — 4.3.2 Monitors: A More Structured Interface to Mutexes
- [hailperin-osm, PDF p. 140](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=140) — 4.5 Condition Variables

### q194 — In an IA-32 system where user and kernel share a page table, what prevents user code from accessing kernel memory?

Intent: Find how an IA-32 system protects kernel memory when user and kernel code share a page table.

Category: `conceptual` · ISA: `x86` · Comparison group: `x86_example`

- [hailperin-osm, PDF p. 307](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=307) — 7.3.1 The Foundation of Protection: Two Processor Modes

### q198 — What do the client stub and server skeleton do during a remote procedure call?

Intent: Understand the responsibilities of client and server middleware during a remote procedure call.

Category: `conceptual` · ISA: `unspecified` · Comparison group: `rpc_stubs`

- [hailperin-osm, PDF p. 472](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=472) — 10.3.1 Principles of Operation for RPC
- [hailperin-osm, PDF p. 473](https://raw.githubusercontent.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction/3567f8f83eafec7c2238271bbd18bea5e0c4f22f/web-site/osm-rev1.3.1.pdf#page=473) — 10.3.1 Principles of Operation for RPC
