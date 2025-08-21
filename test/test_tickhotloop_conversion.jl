# test_tickhotloop_conversion.jl
# Tests the core complexification logic of TickHotLoopF32.jl

# ============================================================================
# TEST FUNCTION
# ============================================================================

@testset "TickHotLoop Conversion Tests" begin
    @testset "apply_quad_phase function" begin
        # Test the core 4-phase rotation logic for the first 5 ticks
        @test apply_quad_phase(1.0f0, 1) ≈ ComplexF32(1.0, 0.0) atol=1e-6
        @test apply_quad_phase(1.0f0, 2) ≈ ComplexF32(0.0, 1.0) atol=1e-6
        @test apply_quad_phase(1.0f0, 3) ≈ ComplexF32(-1.0, 0.0) atol=1e-6
        @test apply_quad_phase(1.0f0, 4) ≈ ComplexF32(0.0, -1.0) atol=1e-6
        @test apply_quad_phase(1.0f0, 5) ≈ ComplexF32(1.0, 0.0) atol=1e-6
    end
end

# ============================================================================
# RUN TEST
# ============================================================================

test_complex_conversion()