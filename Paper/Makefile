# CppSim Makefile
#
# Targets:
#   make test_XXX  — build specific test
#   make tests     — build and run all tests
#   make clean     — remove all build artifacts
#
# Physical dimension:
#   make PHYS_DIM=2   — qubits (default)
#   make PHYS_DIM=3   — qutrits

HIPCC    = hipcc
CXXFLAGS = -O3 -std=c++17 -I./include
LIBS     = -lrocblas -lrocsolver -lm
PHYS_DIM ?= 2
DEFINES  = -DPHYS_DIM=$(PHYS_DIM)

HEADERS = include/types.hpp include/tensor.hpp include/grid.hpp include/workspace.hpp

# ── Tests ─────────────────────────────────────────────────────────────────

test_tensor: tests/test_tensor.cpp include/types.hpp include/tensor.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_tensor tests/test_tensor.cpp $(LIBS)
	@echo "Built tests/test_tensor"

test_contraction: tests/test_contraction.cpp include/types.hpp include/tensor.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_contraction tests/test_contraction.cpp $(LIBS)
	@echo "Built tests/test_contraction"

test_pseudo_sqrt: tests/test_pseudo_sqrt.cpp include/types.hpp include/tensor.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_pseudo_sqrt tests/test_pseudo_sqrt.cpp $(LIBS)
	@echo "Built tests/test_pseudo_sqrt"

test_gate_ops: tests/test_gate_ops.cpp include/types.hpp include/tensor.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_gate_ops tests/test_gate_ops.cpp $(LIBS)
	@echo "Built tests/test_gate_ops"

test_grid: tests/test_grid.cpp include/types.hpp include/tensor.hpp include/grid.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_grid tests/test_grid.cpp $(LIBS)
	@echo "Built tests/test_grid"

test_workspace: tests/test_workspace.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_workspace tests/test_workspace.cpp $(LIBS)
	@echo "Built tests/test_workspace"


test_measures: tests/test_measures.cpp $(HEADERS) include/measures.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_measures tests/test_measures.cpp $(LIBS)
	@echo "Built tests/test_measures"


test_gate: tests/test_gate.cpp $(HEADERS) include/gate.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_gate tests/test_gate.cpp $(LIBS)
	@echo "Built tests/test_gate"

test_entangle:tests/test_entangle.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_entangle tests/test_entangle.cpp $(LIBS)
	@echo "Built tests/test_entagle"

test_3x3_golden: tests/test_3x3_golden.cpp $(HEADERS) include/gate.hpp include/physics.hpp include/measures.hpp
	$(HIPCC) $(CXXFLAGS) -O0 -g $(DEFINES) -o tests/test_3x3_golden tests/test_3x3_golden.cpp $(LIBS)
	@echo "Built tests/test_3x3_golden"


# ── Run all tests ──────────────────────────────────────────────────────────

tests: test_tensor test_contraction test_pseudo_sqrt test_gate_ops test_grid test_workspace  test_measures test_gate test_entangle
	@echo "--- test_tensor ---";      ./tests/test_tensor
	@echo "--- test_contraction ---"; ./tests/test_contraction
	@echo "--- test_pseudo_sqrt ---"; ./tests/test_pseudo_sqrt
	@echo "--- test_gate_ops ---";    ./tests/test_gate_ops
	@echo "--- test_grid ---";        ./tests/test_grid
	@echo "--- test_workspace ---";   ./tests/test_workspace
	@echo "--- test_measures ---";   ./tests/test_measures
	@echo "--- test_gate ---";   ./tests/test_gate
	@echo "--- test_entagle ---";   ./tests/test_entangle

# ── Main simulation ────────────────────────────────────────────────────────

main: main.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main main.cpp $(LIBS)
	@echo "Built main"

# ── Clean ──────────────────────────────────────────────────────────────────

clean:
	rm -f tests/test_tensor tests/test_contraction tests/test_pseudo_sqrt \
	      tests/test_gate_ops tests/test_grid tests/test_workspace \
	      tests/test_measures \
              tests/test_gate main *.o

.PHONY: tests clean
