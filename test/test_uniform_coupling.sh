#!/bin/bash
# test_uniform_coupling.sh — Test bilinear_form_bp with uniform J=1.0 couplings
#
# Hypothesis: heterogeneous couplings cause the normalization mismatch.
# With uniform J=1.0, CppSim and Julia should use identical gates.
#
# Steps:
# 1. Create uniform coupling JSON
# 2. Regenerate CppSim PTMs with uniform J
# 3. Run Julia dump_two_states.jl with uniform JSON
# 4. Run CppSim test → cosine similarity should match Julia's overlap

set -e
CPPSIM=/scratch/MPF/CppSim
JULIA=/scratch/paolod/QC/amd-TensorNetworkQuantumSimulator.jl

echo "═══════════════════════════════════════════════"
echo "  test_uniform_coupling.sh"
echo "═══════════════════════════════════════════════"

# Step 1: Create uniform coupling JSON
echo ""
echo "── Step 1: Create uniform J=1.0 coupling JSON ──────────────"
python3 -c "
import json, copy
with open('$CPPSIM/data/graphs/heavy_hex_3x3.json') as f:
    d = json.load(f)
d_uniform = copy.deepcopy(d)
for e in d_uniform['edges']:
    e['coupling'] = 1.0
with open('$CPPSIM/data/graphs/heavy_hex_3x3_uniform.json','w') as f:
    json.dump(d_uniform, f, indent=2)
print('  Written: heavy_hex_3x3_uniform.json (all couplings = 1.0)')
"

# Step 2: Regenerate CppSim PTMs with uniform J
echo ""
echo "── Step 2: Regenerate CppSim PTMs (uniform J=1.0) ──────────"
python3 build_ptm.py \
    data/heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json \
    data/graphs/heavy_hex_3x3_uniform.json \
    IDEAL_KICK \
    include/heavyhex_ideal_kick_data.hpp
echo "  Rebuilt: heavyhex_ideal_kick_data.hpp (uniform couplings)"

# Step 3: Rebuild CppSim test
echo ""
echo "── Step 3: Rebuild test_inner_op_state ──────────────────────"
rm -f test_inner_op_state
make PHYS_DIM=4 test_inner_op_state 2>&1 | tail -2

# Step 4: Run Julia dump_two_states.jl with uniform JSON
echo ""
echo "── Step 4: Run Julia with uniform couplings ─────────────────"
cat > /tmp/dump_uniform.jl << 'JULIA_EOF'
include("workloads/noisy_ptm_simulation/src/graphs_utils.jl")
include("workloads/noisy_ptm_simulation/src/tnqs_noisy_utils.jl")
include("workloads/noisy_ptm_simulation/src/tnqs_gates.jl")
using TensorNetworkQuantumSimulator
using ITensors

# Use UNIFORM coupling JSON
graph_data = load_graph_json("/scratch/MPF/CppSim/data/graphs/heavy_hex_3x3_uniform.json")
g = graph_data[:g]
obs = collect(g.vertices)[3]

gamma_values = [1.0, 1.4, 1.8]
noise_models = load_scaled_hardware_noise_models("heavy_hex_3x3", gamma_values; data_root="data")

psi_nl0 = paulitensornetworkstate(Float32, v -> v==obs ? "Z" : "I", g)
site_indices = siteinds(psi_nl0)
ptm_tensors = convert_scaled_ptms_to_gates(noise_models, site_indices)

kick_gates = setup_kick_gates(g, 1.41371669)
evolution_gates = setup_hardware_evolution_gates(graph_data, 1, 0.25)
reversed_ev = reverse_evolution_gates(evolution_gates)
push!(reversed_ev, [kick_gates])

apply_kwargs = (; cutoff=1e-4, normalize_tensors=false)

bpc_nl = BeliefPropagationCache(psi_nl0)
gate_its_nl = map(reversed_ev) do gl
    map(gl) do gate_layer
        isa(gate_layer[1], String) ? gate_layer : begin
            built = TensorNetworkQuantumSimulator.toitensor(collect(gate_layer), graph(bpc_nl), siteinds(network(bpc_nl)))
            (;its=[x[1] for x in built], verts=[x[2] for x in built])
        end
    end
end
for subs in gate_its_nl, entry in subs
    if entry isa NamedTuple
        global bpc_nl
        bpc_nl, _ = apply_gates(entry.its, bpc_nl; gate_vertices=entry.verts, apply_kwargs=apply_kwargs, verbose=false)
    end
end

psi_n0 = paulitensornetworkstate(Float32, v -> v==obs ? "Z" : "I", g, site_indices)
bpc_n = BeliefPropagationCache(psi_n0)
ptm_g = ptm_tensors[1.0]
for gl in reversed_ev
    for gate_layer in gl
        global bpc_n
        if isa(gate_layer[1], String)
            model_idx = gate_layer[2]
            (noise_circ, gate_verts) = ptm_g[model_idx]
            bpc_n, _ = apply_gates(reverse(noise_circ), bpc_n; gate_vertices=reverse(gate_verts), apply_kwargs=apply_kwargs, verbose=false)
        else
            bpc_n, _ = apply_gates(gate_layer, bpc_n; apply_kwargs=apply_kwargs, verbose=false)
        end
    end
end

ov = inner(network(bpc_nl), network(bpc_n); alg="bp", cache_update_kwargs=apply_kwargs)
nl2 = norm_sqr(network(bpc_nl); alg="bp", cache_update_kwargs=(;cutoff=1e-4))
n2  = norm_sqr(network(bpc_n);  alg="bp", cache_update_kwargs=(;cutoff=1e-4))
println("JULIA_UNIFORM_OVERLAP = $ov")
println("JULIA_NL_NORM2 = $nl2")
println("JULIA_N_NORM2  = $n2")
println("JULIA_COSINE   = $(ov / sqrt(nl2*n2))")
JULIA_EOF

cd $JULIA
julia --project=. /tmp/dump_uniform.jl 2>&1 | grep -E "JULIA|ERROR" | head -10
cd $CPPSIM

# Step 5: Run CppSim test
echo ""
echo "── Step 5: Run CppSim test ──────────────────────────────────"
./test_inner_op_state 2>/dev/null | grep -E "bilinear|cosine|PASS|FAIL|Results"

# Step 6: Restore original PTMs
echo ""
echo "── Step 6: Restore original PTMs ───────────────────────────"
python3 build_ptm.py \
    data/heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json \
    data/graphs/heavy_hex_3x3.json \
    IDEAL_KICK \
    include/heavyhex_ideal_kick_data.hpp
echo "  Restored: heavyhex_ideal_kick_data.hpp (original couplings)"
