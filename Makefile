# CppSim Makefile
#
# Build from repository root:
#   make PHYS_DIM=4 main_heavyhex_gamma_mpf_2gpu
#   make PHYS_DIM=4 main_lightcone
#   make PHYS_DIM=4 main_heavyhex_gamma_mpf_DG_2gpu
#   make PHYS_DIM=4 main_gauge_compare
#
# Physical dimension (always use 4 for heavy-hex):
#   make PHYS_DIM=4  — Pauli basis (D=4), required for all heavy-hex binaries

HIPCC    = hipcc
CXXFLAGS = -O3 -std=c++17 -I./include
LIBS     = -lrocblas -lrocsolver -lm
PHYS_DIM ?= 4
DEFINES  = -DPHYS_DIM=$(PHYS_DIM)

SRC = examples
HEADERS = include/types.hpp include/tensor.hpp include/grid.hpp include/workspace.hpp

# ── Primary binaries ───────────────────────────────────────────────────────────

DEPS_2GPU = $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/bilinear_form_bp_gpu.hpp include/bilinear_form_bp_gpu3.hpp \
	include/overlap_gpu.hpp

main_heavyhex_gamma_mpf_2gpu: $(SRC)/main_heavyhex_gamma_mpf_2gpu.cpp $(DEPS_2GPU) \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS) -lpthread
	@echo "Built $@"

main_heavyhex_ergodic_mpf_2gpu: $(SRC)/main_heavyhex_gamma_mpf_2gpu.cpp $(DEPS_2GPU) \
	include/heavyhex_ideal_kick_eps010_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps015_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps020_gamma1_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DERGODIC -o $@ $(SRC)/main_heavyhex_gamma_mpf_2gpu.cpp $(LIBS) -lpthread
	@echo "Built $@"

main_heavyhex_ergodic2_mpf_2gpu: $(SRC)/main_heavyhex_gamma_mpf_2gpu.cpp $(DEPS_2GPU) \
	include/heavyhex_ideal_kick_eps010_gamma2_data.hpp \
	include/heavyhex_ideal_kick_eps015_gamma2_data.hpp \
	include/heavyhex_ideal_kick_eps020_gamma2_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DERGODIC2 -o $@ $(SRC)/main_heavyhex_gamma_mpf_2gpu.cpp $(LIBS) -lpthread
	@echo "Built $@"

main_heavyhex_gamma_mpf_DG_2gpu: $(SRC)/main_heavyhex_gamma_mpf_DG_2gpu.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp include/diag_gauge_ip.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) $(DFLAGS) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@"

main_pauli_sweep: $(SRC)/main_pauli_sweep.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp include/trotter.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@ (heavy_hex_3x3)"

main_pauli_sweep_rect7x7: $(SRC)/main_pauli_sweep.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp include/trotter.hpp \
	include/rect_7x7.hpp \
	include/rect_7x7_ideal_kick_eps010_data.hpp \
	include/rect_7x7_noisy_eps010_gamma1_data.hpp \
	include/rect_7x7_noisy_eps010_gamma2_data.hpp \
	include/rect_7x7_noisy_eps010_gamma3_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DGRAPH_rect_7x7 -o $@ $(SRC)/main_pauli_sweep.cpp $(LIBS)
	@echo "Built $@ (rect_7x7)"

main_generic_lightcone: $(SRC)/main_generic_lightcone.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp include/trotter.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) $(DFLAGS) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@ (heavy_hex_3x3)"

main_generic_lightcone_rect7x7: $(SRC)/main_generic_lightcone.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp include/trotter.hpp \
	include/rect_7x7.hpp include/rect_7x7_ideal_kick_eps010_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DGRAPH_rect_7x7 -o $@ $(SRC)/main_generic_lightcone.cpp $(LIBS)
	@echo "Built $@ (rect_7x7)"

main_lightcone: $(SRC)/main_lightcone.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@"

main_gauge_compare: $(SRC)/main_gauge_compare.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp \
	include/bilinear_form_blf_gpu2.hpp \
	include/sqrt_gauge_ip.hpp \
	include/diag_gauge_ip.hpp \
	include/sqrt_bp_ip.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS) -llapack -lblas
	@echo "Built $@"

# ── Legacy/experimental ────────────────────────────────────────────────────────

main_heavyhex: $(SRC)/main_heavyhex.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@"

main_heavyhex_mpf: $(SRC)/main_heavyhex_mpf.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp include/overlap_gpu.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS) -lpthread
	@echo "Built $@"

main_heavyhex_gamma_mpf: $(SRC)/main_heavyhex_gamma_mpf.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@"

main_heavyhex_gamma_mpf_ip: $(SRC)/main_heavyhex_gamma_mpf_ip.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/heavyhex_noisy_eps010_gamma2_data.hpp \
	include/heavyhex_noisy_eps010_gamma3_data.hpp \
	include/bilinear_form_bp.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@"

main_heavyhex_ergodic_mpf: $(SRC)/main_heavyhex_ergodic_mpf.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_ideal_kick_eps010_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps015_gamma1_data.hpp \
	include/heavyhex_ideal_kick_eps020_gamma1_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o $@ $(SRC)/$@.cpp $(LIBS)
	@echo "Built $@"

# ── Tests ──────────────────────────────────────────────────────────────────────

test_blf_sgemm: test/test_blf_sgemm.cpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_rocsolver_syevd: test/test_rocsolver_syevd.cpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_blf_gpu: test/test_blf_gpu.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/bilinear_form_bp.hpp include/bilinear_form_bp_gpu.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_blf_gpu3: test/test_blf_gpu3.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/bilinear_form_bp.hpp \
	include/bilinear_form_bp_cpu2.hpp \
	include/bilinear_form_bp_gpu3.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_blf_cpu2: test/test_blf_cpu2.cpp $(HEADERS) include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/heavyhex_ideal_kick_eps010_data.hpp \
	include/heavyhex_noisy_eps010_gamma1_data.hpp \
	include/bilinear_form_bp.hpp include/bilinear_form_bp_cpu2.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_rect7x7_trotter: test/test_rect7x7_trotter.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/rect_7x7.hpp include/rect_7x7_ideal_kick_eps010_data.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DPHYS_DIM=4 -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_rect7x7: test/test_rect7x7.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/rect_7x7.hpp include/rect_7x7_ideal_kick_eps010_data.hpp \
	include/trotter.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -DPHYS_DIM=4 -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_heavyhex: test/test_heavyhex.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/gate.hpp include/measures.hpp include/workspace.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

test_overlap_gpu: test/test_overlap_gpu.cpp $(HEADERS) \
	include/heavyhex.hpp include/heavyhex_physics.hpp \
	include/overlap_gpu.hpp include/measures.hpp
	$(HIPCC) $(CXXFLAGS) $(DEFINES) -o test/$@ test/$@.cpp $(LIBS)
	@echo "Built test/$@"

# ── Clean ──────────────────────────────────────────────────────────────────────

clean:
	rm -f main_heavyhex_gamma_mpf_2gpu main_heavyhex_gamma_mpf_DG_2gpu \
	      main_heavyhex_ergodic_mpf_2gpu main_heavyhex_ergodic2_mpf_2gpu \
	      main_lightcone main_gauge_compare \
	      main_heavyhex main_heavyhex_mpf main_heavyhex_gamma_mpf \
	      main_heavyhex_gamma_mpf_ip main_heavyhex_ergodic_mpf \
	      test/test_blf_* test/test_rocsolver_* test/test_heavyhex \
	      test/test_overlap_* *.o

.PHONY: clean
