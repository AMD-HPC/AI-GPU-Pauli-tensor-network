"""
dump_two_states.jl — Dump two evolved states + their overlap for double-layer BP validation

Evolves noiseless and noisy(γ=1.0) for 1 cycle, computes the TRUE two-state
overlap ⟨ψ_nl|ψ_noisy⟩ via BP, and dumps both ψ tensors.

This gives the target for the Python double-layer BP implementation.

Output:
  - prints inner(ψ_nl, ψ_noisy; alg="bp")  ← THE TARGET
  - dumps both states' tensors to two_states_dump.txt
"""

include("workloads/noisy_ptm_simulation/src/graphs_utils.jl")
include("workloads/noisy_ptm_simulation/src/tnqs_noisy_utils.jl")
include("workloads/noisy_ptm_simulation/src/tnqs_gates.jl")
using TensorNetworkQuantumSimulator
using ITensors

graph_data = load_graph_json("data/graphs/heavy_hex_3x3.json")
g = graph_data[:g]
obs = collect(g.vertices)[3]

# Noise models
gamma_values = [1.0, 1.4, 1.8]
noise_models = load_scaled_hardware_noise_models("heavy_hex_3x3", gamma_values; data_root="data")

# Build noiseless and noisy initial states with SHARED site indices
psi_nl0 = paulitensornetworkstate(Float32, v -> v==obs ? "Z" : "I", g)
site_indices = siteinds(psi_nl0)
ptm_tensors = convert_scaled_ptms_to_gates(noise_models, site_indices)

kick_gates = setup_kick_gates(g, 1.41371669)
evolution_gates = setup_hardware_evolution_gates(graph_data, 1, 0.25)
reversed_ev = reverse_evolution_gates(evolution_gates)
push!(reversed_ev, [kick_gates])

apply_kwargs = (; cutoff=1e-4, normalize_tensors=false)

# Evolve noiseless
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

# Evolve noisy γ=1.0 (with noise layers)
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

# ── THE TARGET: true two-state overlap ─────────────────────────────────────
ov = inner(network(bpc_nl), network(bpc_n); alg="bp", cache_update_kwargs=apply_kwargs)
println("═══════════════════════════════════════════════")
println("TARGET: inner(ψ_nl, ψ_noisy; alg=bp) = $ov")
println("═══════════════════════════════════════════════")

# ── Dump both states ───────────────────────────────────────────────────────
function dump_state(bpc, label, fh)
    net = network(bpc)
    for v in sort(collect(vertices(net)))
        t = net[v]
        is = inds(t)
        pauli = only(filter(i->hastags(i,"Pauli"), is))
        links = filter(i->hastags(i,"Link,u"), is)
        # Order: links..., pauli
        arr = ITensors.array(t, links..., pauli)
        sz = size(arr)
        vflat = vec(arr)
        print(fh, "$label $(v[1]) ndim $(length(sz)) shape")
        for s in sz; print(fh, " $s"); end
        print(fh, " data")
        for x in vflat; print(fh, " $(Float64(x))"); end
        println(fh)
    end
end

open("two_states_dump.txt", "w") do fh
    println(fh, "OVERLAP $ov")
    dump_state(bpc_nl, "NL", fh)
    dump_state(bpc_n, "N", fh)
end
println("Dumped to two_states_dump.txt")
