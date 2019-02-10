using ParallelMCWF
using Test
using QuantumOptics

@testset "ParallelMCWF.jl" begin
    # Write your own tests here.
    Δ = 1.; F = 2.; γ = 1.;
    b = FockBasis(10);
    N = number(b);
    a = destroy(b);
    H = -Δ*N + F*(a + dagger(a));
    J = [sqrt(γ) * a];
    T = Float64[0.:0.1:10.;]

    # Initial conditions
    Ψ₀ = fockstate(b,0);

    # Test multithreaded_mcwf
    tout, Ψt = multithreaded_mcwf(T, Ψ₀, H, J, 100; seed=UInt(1), reltol=1e-7);
    tout2, Ψt2 = multithreaded_mcwf(T, Ψ₀, H, J, 100; seed=UInt(1), reltol=1e-7);
    @test Ψt == Ψt2

    tout, Ψt = multithreaded_mcwf(T, Ψ₀, H, J, 2; seed=UInt(1), reltol=1e-7)
    tout, Ψt2 = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7)
    @test Ψt[1] == Ψt2
end

Δ = 1.; F = 2.; γ = 1.;
b = FockBasis(10);
N = number(b);
a = destroy(b);
H = -Δ*N + F*(a + dagger(a));
J = [sqrt(γ) * a];
T = collect(0:10)

# Initial conditions
Ψ₀ = fockstate(b,0);
sol = pmcwf(T, Ψ₀, H, J;Ntrajectories=3, parallel_type=:threads, seed=UInt(1), reltol=1e-7)
sol = pmcwf(T, Ψ₀, H, J;Ntrajectories=2, parallel_type=:threads, reltol=1e-7, seed=UInt(1))
sol[1] == sol[2]
sol = pmcwf(T, Ψ₀, H, J;Ntrajectories=2, parallel_type=:none, reltol=1e-7, seed=UInt(1))

multithreaded_mcwf(T, Ψ₀, H, J, 2; seed=UInt(1), fout=(t,x)->expect(N,x)/norm(x)^2,reltol=1e-7)
tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7)
out_type = eltype(timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7)[2])
typeof(timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7))
sol = (Vector{Float64}(),Vector{out_type}())
multithreaded_mcwf(T, Ψ₀, H, J, 2; seed=UInt(1), reltol=1e-7)
