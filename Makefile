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

test_3x3_golden: tests/test_3x3_golden.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_3x3_golden tests/test_3x3_golden.cpp $(LIBS)
	@echo "Built tests/test_3x3_golden"




test_measures: tests/test_measures.cpp $(HEADERS) include/measures.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_measures tests/test_measures.cpp $(LIBS)
	@echo "Built tests/test_measures"


test_gate: tests/test_gate.cpp $(HEADERS) include/gate.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_gate tests/test_gate.cpp $(LIBS)
	@echo "Built tests/test_gate"

test_entangle:tests/test_entangle.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o tests/test_entangle tests/test_entangle.cpp $(LIBS)
	@echo "Built tests/test_entagle"


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
	@echo "--- test_3x3_golden ---";   ./tests/test_3x3_golden

test_heavyhex: test/test_heavyhex.cpp include/types.hpp include/heavyhex.hpp include/heavyhex_physics.hpp include/gate.hpp include/measures.hpp include/workspace.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/test_heavyhex test/test_heavyhex.cpp $(LIBS)
	@echo "Built test/test_heavyhex"

test_noisy_heavyhex: test/test_noisy_heavyhex.cpp include/types.hpp include/heavyhex.hpp include/heavyhex_physics.hpp include/heavyhex_noise_data.hpp include/gate.hpp include/measures.hpp include/workspace.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/test_noisy_heavyhex test/test_noisy_heavyhex.cpp $(LIBS)
	@echo "Built test/test_noisy_heavyhex"

test_inner_op_state: test_inner_op_state.cpp include/types.hpp include/heavyhex.hpp include/heavyhex_physics.hpp include/heavyhex_ideal_kick_data.hpp include/gate.hpp include/measures.hpp include/workspace.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test_inner_op_state test_inner_op_state.cpp $(LIBS)
	@echo "Built test_inner_op_state"

# ── Main simulation ────────────────────────────────────────────────────────

main: main.cpp $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main main.cpp $(LIBS)
	@echo "Built main"

main_heavyhex: main_heavyhex.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex main_heavyhex.cpp $(LIBS)
	@echo "Built main_heavyhex"

main_heavyhex_noisy: main_heavyhex_noisy.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp include/heavyhex_noise_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_noisy main_heavyhex_noisy.cpp $(LIBS)
	@echo "Built main_heavyhex_noisy"

main_heavyhex_mpf: main_heavyhex_mpf.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp include/heavyhex_noise_data.hpp include/overlap_gpu.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_mpf main_heavyhex_mpf.cpp $(LIBS) -lpthread
	@echo "Built main_heavyhex_mpf"

main_heavyhex_mpf2: main_heavyhex_mpf2.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp include/bilinear_form_bp.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_mpf2 main_heavyhex_mpf2.cpp $(LIBS) -lpthread
	@echo "Built main_heavyhex_mpf2"

DEPS_2GPU = $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/bilinear_form_bp_gpu.hpp include/bilinear_form_bp_gpu3.hpp \
	include/overlap_gpu.hpp

main_heavyhex_gamma_mpf_2gpu: main_heavyhex_gamma_mpf_2gpu.cpp $(DEPS_2GPU) \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_gamma_mpf_2gpu main_heavyhex_gamma_mpf_2gpu.cpp $(LIBS) -lpthread
	@echo "Built main_heavyhex_gamma_mpf_2gpu"

main_heavyhex_ergodic_mpf_2gpu: main_heavyhex_gamma_mpf_2gpu.cpp $(DEPS_2GPU) \
	include/heavyhex_ideal_kick_eps010_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps015_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps020_gamma1_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DERGODIC -o main_heavyhex_ergodic_mpf_2gpu main_heavyhex_gamma_mpf_2gpu.cpp $(LIBS) -lpthread
	@echo "Built main_heavyhex_ergodic_mpf_2gpu"

main_heavyhex_ergodic2_mpf_2gpu: main_heavyhex_gamma_mpf_2gpu.cpp $(DEPS_2GPU) \
	include/heavyhex_ideal_kick_eps010_gamma2_data.hpp \
	include/heavyhex_ideal_kick_eps015_gamma2_data.hpp \
	include/heavyhex_ideal_kick_eps020_gamma2_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DERGODIC2 -o main_heavyhex_ergodic2_mpf_2gpu main_heavyhex_gamma_mpf_2gpu.cpp $(LIBS) -lpthread
	@echo "Built main_heavyhex_ergodic2_mpf_2gpu"

main_lightcone: main_lightcone.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_lightcone main_lightcone.cpp $(LIBS)
	@echo "Built main_lightcone"

main_heavyhex_ergodic_mpf: main_heavyhex_ergodic_mpf.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_ideal_kick_eps010_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps015_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps020_gamma1_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_ergodic_mpf main_heavyhex_ergodic_mpf.cpp $(LIBS)
	@echo "Built main_heavyhex_ergodic_mpf"

main_heavyhex_gamma_mpf_ip: main_heavyhex_gamma_mpf_ip.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp \
	include/bilinear_form_bp.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_gamma_mpf_ip main_heavyhex_gamma_mpf_ip.cpp $(LIBS)
	@echo "Built main_heavyhex_gamma_mpf_ip"

test_blf_gpu3: test_blf_gpu3.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/bilinear_form_bp.hpp \
	include/bilinear_form_bp_cpu2.hpp \
	include/bilinear_form_bp_gpu3.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test_blf_gpu3 test_blf_gpu3.cpp $(LIBS)
	@echo "Built test_blf_gpu3"

test_blf_sgemm: test_blf_sgemm.cpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test_blf_sgemm test_blf_sgemm.cpp $(LIBS)
	@echo "Built test_blf_sgemm"

test_sqrt_gauge_ip: test_sqrt_gauge_ip.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/bilinear_form_blf_gpu2.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DPHYS_DIM=4 -o test_sqrt_gauge_ip test_sqrt_gauge_ip.cpp $(LIBS)
	@echo "Built test_sqrt_gauge_ip"

main_gauge_compare: main_gauge_compare.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp \
	include/bilinear_form_blf_gpu2.hpp \
	include/sqrt_gauge_ip.hpp \
	include/diag_gauge_ip.hpp \
	include/sqrt_bp_ip.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DPHYS_DIM=4 -o main_gauge_compare main_gauge_compare.cpp $(LIBS)
	@echo "Built main_gauge_compare"

test_blf_cpu2: test_blf_cpu2.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/bilinear_form_bp.hpp \
	include/bilinear_form_bp_cpu2.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test_blf_cpu2 test_blf_cpu2.cpp $(LIBS)
	@echo "Built test_blf_cpu2"

test_blf_gpu: test_blf_gpu.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/bilinear_form_bp.hpp \
	include/bilinear_form_bp_gpu.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test_blf_gpu test_blf_gpu.cpp $(LIBS)
	@echo "Built test_blf_gpu"

main_heavyhex_gamma_mpf: main_heavyhex_gamma_mpf.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_gamma_mpf main_heavyhex_gamma_mpf.cpp $(LIBS)
	@echo "Built main_heavyhex_gamma_mpf"

main_heavyhex_eps_mpf: main_heavyhex_eps_mpf.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_data.hpp \
	include/heavyhex_noisy_eps015_data.hpp \
	include/heavyhex_noisy_eps020_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o main_heavyhex_eps_mpf main_heavyhex_eps_mpf.cpp $(LIBS)
	@echo "Built main_heavyhex_eps_mpf"

test_overlap_cpu: test/test_overlap_cpu.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp include/heavyhex_noise_data.hpp include/overlap_cpu.hpp include/measures.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/test_overlap_cpu test/test_overlap_cpu.cpp $(LIBS) -lopenblas
	@echo "Built test/test_overlap_cpu"

test_overlap_gpu: test/test_overlap_gpu.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp include/heavyhex_noise_data.hpp include/overlap_gpu.hpp include/measures.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/test_overlap_gpu test/test_overlap_gpu.cpp $(LIBS)
	@echo "Built test/test_overlap_gpu"

# ── Clean ──────────────────────────────────────────────────────────────────

clean:
	rm -f tests/test_tensor tests/test_contraction tests/test_pseudo_sqrt \
	      tests/test_gate_ops tests/test_grid tests/test_workspace \
	      tests/test_measures tests/test_3x3_golden\
              tests/test_gate main *.o

.PHONY: tests clean
