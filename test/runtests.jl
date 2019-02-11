using ParallelMCWF
using Test
using QuantumOptics

@testset "ParallelMCWF.jl" begin
    # Model: driven-dissipative harmonic oscillator
    Δ = 1.; F = 2.; γ = 1.;
    b = FockBasis(10);
    N = number(b);
    a = destroy(b);
    H = -Δ*N + F*(a + dagger(a));
    J = [sqrt(γ) * a];
    T = Float64[0.:0.1:10.;]
	Ntrajectories = 500;

    # Initial conditions
    Ψ₀ = fockstate(b,0);

	t, ρ = timeevolution.master(T,Ψ₀,H,J)

	# Test pmcwf and kets_to_dm
	t, ψ_none = pmcwf(T, Ψ₀, H, [sqrt(γ)*a]; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:none)
	ρ_none = kets_to_dm([ψ_none[i][end] for i in 1:length(ψ_none)];parallel_type=:none)
	err_none =  tracedistance(ρ[end], ρ_none)
	@test err_none < 1e-2

	t, ψ_pmap = pmcwf(T, Ψ₀, H, [sqrt(γ)*a]; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:pmap)
	ρ_pmap = kets_to_dm([ψ_pmap[i][end] for i in 1:length(ψ_pmap)];parallel_type=:none)
	err_pmap =  tracedistance(ρ[end], ρ_pmap)
	@test err_pmap < 1e-2

	t, ψ_threads = pmcwf(T, Ψ₀, H, [sqrt(γ)*a]; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:threads)
	ρ_threads = kets_to_dm([ψ_threads[i][end] for i in 1:length(ψ_threads)];parallel_type=:none)
	err_threads =  tracedistance(ρ[end], ρ_threads)
	@test err_threads < 1e-2

	# Test kets_to_obs
	O = randoperator(b); O = O*dagger(O) + one(b);
	O_ref = expect(O,ρ[end])

	ψ = [ψ_none[i][end] for i in 1:length(ψ_none)];
	O_none = kets_to_obs(O,ψ;parallel_type=:none)
	@test abs((O_none - O_ref)/O_ref) < 1e-2

	O_pmap = kets_to_obs(O,ψ;parallel_type=:pmap)
	@test abs((O_pmap - O_ref)/O_ref) < 1e-2

	O_threads = kets_to_obs(O,ψ;parallel_type=:threads)
	@test abs((O_pmap - O_ref)/O_ref) < 1e-2

	# TO DO: add tests on extended models
	# TO DO: test functions of trajs_IO.jl
end
