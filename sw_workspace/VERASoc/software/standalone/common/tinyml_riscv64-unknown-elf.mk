#ifneq (, $(shell which riscv-none-embed-gcc))
#RISCV_BIN ?= riscv32-unknown-elf-
RISCV_BIN ?= riscv-none-embed-
#else
#ifneq (, $(shell which riscv64-unknown-elf-gcc))
#  RISCV_BIN ?= riscv64-unknown-elf-
#else
#$(warning No RISC-V toolchain detected, please install riscv-none-embed- from xpack)
#endif
#endif

RISCV_CC=${RISCV_BIN}gcc
RISCV_CXX=${RISCV_BIN}g++
RISCV_OBJCOPY=${RISCV_BIN}objcopy
RISCV_OBJDUMP=${RISCV_BIN}objdump

MARCH := rv32i
MABI := ilp32
BENCH ?= no
DEBUG?=no
DEBUG_Og?=no

ifeq ($(RV_M),yes)
	MARCH := $(MARCH)m
endif

ifeq ($(RV_A),yes)
	MARCH := $(MARCH)a
else
    ifeq ($(RV_C),yes)
	    MARCH := $(MARCH)a
    endif
endif

ifeq ($(RV_F),yes)
	MARCH := $(MARCH)f

    ifeq ($(RV_D),no)
	    MABI := $(MABI)f
    endif
endif

ifeq ($(RV_D),yes)
	MARCH := $(MARCH)d
	MABI := $(MABI)d
endif

ifeq ($(RV_C),yes)
	MARCH := $(MARCH)c
endif

#MARCH := $(MARCH)xcustom

ifeq ($(DEBUG),yes)
    ifneq ($(DEBUG_OG),yes)
	    CFLAGS += -g3 -O0
    else
	    CFLAGS += -g3 -Og
    endif
endif

ifneq ($(DEBUG),yes)
    ifneq ($(BENCH),yes)
	    CFLAGS += -Os
    else
	    CFLAGS += -O3
    endif
endif

CFLAGS += -march=$(MARCH) -mabi=$(MABI) -DUSE_GP
LDFLAGS += -march=$(MARCH) -mabi=$(MABI)

