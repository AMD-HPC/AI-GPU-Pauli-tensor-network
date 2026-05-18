/*
 * tests/test_tensor.cpp
 *
 * Test: Tensor creation, shape, matrix view, copy to/from host.
 *
 * Build:
 *   hipcc -O2 -std=c++17 -I../include -o test_tensor test_tensor.cpp
 * Run:
 *   ./test_tensor
 */

#include "types.hpp"
#include "tensor.hpp"
#include <cstdio>
#include <cassert>
#include <cmath>

int main() {
    printf("=== test_tensor ===\n");

    // Test 1: interior site tensor shape [D, chi, chi, chi, chi]
    {
        int chi = 10;
        Tensor<float> t({D, chi, chi, chi, chi});
        t.print_shape("interior_site");
        assert(t.total == D * chi*chi*chi*chi);
        assert(t.ndim == 5);

        // Matrix view: left=1 (phys only) → rows=D, cols=chi^4
        auto [m, n] = t.matrix_dims(1);
        printf("  matrix_dims(1): m=%d n=%d (D=%d, chi^4=%d)\n", m, n, D, chi*chi*chi*chi);
        assert(m == D);
        assert(n == chi*chi*chi*chi);

        // Matrix view: left=4 → rows=D*chi^3, cols=chi (for QR split)
        auto [m2, n2] = t.matrix_dims(4);
        printf("  matrix_dims(4): m=%d n=%d (D*chi^3=%d, chi=%d)\n", m2, n2, D*chi*chi*chi, chi);
        assert(m2 == D*chi*chi*chi);
        assert(n2 == chi);
    }

    // Test 2: edge site tensor [D, chi, chi, chi]
    {
        int chi = 10;
        Tensor<float> t({D, chi, chi, chi});
        t.print_shape("edge_site");
        assert(t.total == D * chi*chi*chi);
    }

    // Test 3: corner site tensor [D, chi, chi]
    {
        int chi = 10;
        Tensor<float> t({D, chi, chi});
        t.print_shape("corner_site");
        assert(t.total == D * chi*chi);
    }

    // Test 4: copy to/from host
    {
        int chi = 4;
        Tensor<float> t({D, chi, chi});
        // Fill with known values on host
        std::vector<float> h(D * chi * chi);
        for (int i = 0; i < (int)h.size(); i++) h[i] = (float)i;
        t.copy_from_host(h);

        // Read back and verify
        auto h2 = t.copy_to_host();
        for (int i = 0; i < (int)h.size(); i++) {
            assert(std::abs(h2[i] - h[i]) < 1e-6f);
        }
        printf("  copy_to/from_host: OK\n");
        t.print_elements(8, "corner_chi4");
    }

    // Test 5: env tensor [chi, chi] (BP message / sqrt_env)
    {
        int chi = 50;
        Tensor<float> msg({chi, chi});
        msg.print_shape("BP_message");
        assert(msg.total == chi*chi);
        printf("  BP message %.2f KB\n", msg.bytes()/1e3);
    }

    // Test 6: MatrixView
    {
        int chi = 10;
        Tensor<float> t({D, chi, chi, chi, chi});  // interior site
        // QR split: left=4 dims → m=D*chi^3, n=chi
        MatrixView<float> mv(t, 4);
        printf("  MatrixView: m=%d n=%d ld=%d\n", mv.m, mv.n, mv.ld);
        assert(mv.m == D*chi*chi*chi);
        assert(mv.n == chi);
        assert(mv.ld == mv.m);
    }

    // Test 7: steady-state sizes (chi=50)
    {
        int chi = 50;
        printf("\n  --- Steady state memory (chi=%d, D=%d) ---\n", chi, D);
        Tensor<float> interior({D, chi, chi, chi, chi});
        interior.print_shape("interior (4 bonds)");
        Tensor<float> edge({D, chi, chi, chi});
        edge.print_shape("edge     (3 bonds)");
        Tensor<float> corner({D, chi, chi});
        corner.print_shape("corner   (2 bonds)");

        // QR split of interior: left=[D,chi,chi,chi], right=[chi]
        // m = D*chi^3 = 2*125000 = 250000, n = chi = 50
        // But wait — we split along bond_v1v2:
        // ψᵥ₁ after contracting envs: [D, chi_env1, chi_env2, chi_env3, chi_bond_v1v2]
        // left_inds = all except bond_v1v2 → m = D*chi^3
        // right_inds = bond_v1v2 → n = chi
        // This matches Julia: QR1: 125000×200 (D*chi^3 × chi*D = 50^3 * 2 × 50 * 4 = ?)
        // Actually n = chi * D because right_inds = [bond_v1v2, phys] ?
        // Let's check: from logs QR1: 125000×200 → m=125000=50^3*2/2... 
        // 125000 = 50^3 = chi^3, 200 = chi*D = 50*4... hmm D=4 here?
        // No: 125000 = chi^3 = 50^3, n=200=chi*D? No, D=2 so chi*D=100
        // 200 = 4*chi? No. 200 = chi*4 where 4=D*2? 
        // Actually: after contracting envs, right=[phys, bond] → n=D*chi=2*50=100? 
        // But log says 200... let me just print what we get
        // QR split matching Julia: left=[chi,chi,chi], right=[D, chi_bond_v1v2]
        // m = chi^3 = 125000, n = D*chi = 200
        // Shape after env contraction: [chi_UP, chi_DOWN, chi_LEFT, D, chi_bond_v1v2]
        // left = first 3 dims = bond dims without phys
        // right = last 2 dims = [D, bond_v1v2]
        auto [m3, n3] = Tensor<float>({chi,chi,chi,D,chi}).matrix_dims(3);
        printf("  Interior QR split (correct): m=%d n=%d  (chi^3=%d, D*chi=%d)\n",
            m3, n3, chi*chi*chi, D*chi);
        assert(m3 == chi*chi*chi);   // 125000
        assert(n3 == D*chi);         // 200 = 2*50
    }

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
