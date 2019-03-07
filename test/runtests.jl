using ParallelMCWF
using Test
using QuantumOptics

@testset "ParallelMCWF.jl" begin
	# Define parameters for spin coupled to electric field mode.
	ωc = 1.2
	ωa = 0.9
	g = 1.0
	γ = 0.5
	κ = 1.1

	Ntrajectories = 500
	T = Float64[0.:0.1:10.;]

	# Define operators
	fockbasis = FockBasis(8)
	spinbasis = SpinBasis(1//2)
	basis = tensor(spinbasis, fockbasis)

	sx = sigmax(spinbasis)
	sy = sigmay(spinbasis)
	sz = sigmaz(spinbasis)
	sp = sigmap(spinbasis)
	sm = sigmam(spinbasis)

	# Hamiltonian
	Ha = embed(basis, 1, 0.5*ωa*sz)
	Hc = embed(basis, 2, ωc*number(fockbasis))
	Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
	H = Ha + Hc + Hint
	Hdense = dense(H)
	Hlazy = LazySum(Ha, Hc, Hint)

	# Jump operators
	Ja = embed(basis, 1, sqrt(γ)*sm)
	Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
	J = [Ja, Jc]
	Jdense = map(dense, J)
	Jlazy = [LazyTensor(basis, 1, sqrt(γ)*sm), LazyTensor(basis, 2, sqrt(κ)*destroy(fockbasis))]

	# Initial conditions
	Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)
	ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)

	##
	# Check seeding

	@inline function allequal(x)
	    length(x) < 2 && return true
	    e1 = x[1]
	    i = 2
	    @inbounds for i=2:length(x)
	        x[i] == e1 || return false
	    end
	    return true
	end
	t, Ψ_none = pmcwf(T, Ψ₀, H, J; seed=UInt(1), Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:none);
	@test allequal(Ψ_none)

	t, Ψ_pmap = pmcwf(T, Ψ₀, H, J; seed=UInt(1), Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:pmap);
	@test allequal(Ψ_pmap)

	t, Ψ_threads = pmcwf(T, Ψ₀, H, J; seed=UInt(1), Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:threads);
	@test allequal(Ψ_threads)

	t, Ψ_parfor = pmcwf(T, Ψ₀, H, J; seed=UInt(1), Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:parfor);
	@test allequal(Ψ_parfor)

	t, Ψ_sthreads = pmcwf(T, Ψ₀, H, J; seed=UInt(1), Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:split_threads);
	@test allequal(Ψ_sthreads)


	##
	# Check convergence to the timeevolution.master-evolved solution

	t, ρ = timeevolution.master(T,Ψ₀,H,J);

	# Test pmcwf and kets_to_dm
	t, Ψ_none = pmcwf(T, Ψ₀, H, J; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:none);
	ρ_none = kets_to_dm([Ψ_none[i][end] for i in 1:length(Ψ_none)];parallel_type=:none);
	err_none =  tracedistance(ρ[end], ρ_none);
	@test err_none < 1e-2

	t, Ψ_pmap = pmcwf(T, Ψ₀, H, J; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:pmap);
	ρ_pmap = kets_to_dm([Ψ_pmap[i][end] for i in 1:length(Ψ_pmap)];parallel_type=:none);
	err_pmap =  tracedistance(ρ[end], ρ_pmap);
	@test err_pmap < 1e-2

	t, Ψ_threads = pmcwf(T, Ψ₀, H, J; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:threads);
	ρ_threads = kets_to_dm([Ψ_threads[i][end] for i in 1:length(Ψ_threads)];parallel_type=:none);
	err_threads =  tracedistance(ρ[end], ρ_threads);
	@test err_threads < 1e-2

	t, Ψ_parfor = pmcwf(T, Ψ₀, H, J; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:parfor);
	ρ_parfor = kets_to_dm([Ψ_parfor[i][end] for i in 1:length(Ψ_parfor)];parallel_type=:none);
	err_parfor =  tracedistance(ρ[end], ρ_parfor);
	@test err_parfor < 1e-2

	t, Ψ_sthreads = pmcwf(T, Ψ₀, H, J; Ntrajectories=Ntrajectories, progressbar=false, parallel_type=:split_threads);
	ρ_sthreads = kets_to_dm([Ψ_sthreads[i][end] for i in 1:length(Ψ_sthreads)];parallel_type=:none);
	err_sthreads =  tracedistance(ρ[end], ρ_sthreads);
	@test err_sthreads < 1e-2

	##
	# Test kets_to_obs

	O = randoperator(basis); O = O*dagger(O) + one(basis);
	O_ref = expect(O,ρ[end])

	Ψ = [Ψ_none[i][end] for i in 1:length(Ψ_none)];
	O_none = kets_to_obs(O,Ψ;parallel_type=:none)
	@test abs((O_none - O_ref)/O_ref) < 1e-2

	O_pmap = kets_to_obs(O,Ψ;parallel_type=:pmap);
	@test abs((O_pmap - O_ref)/O_ref) < 1e-2

	O_threads = kets_to_obs(O,Ψ;parallel_type=:threads);
	@test abs((O_threads - O_ref)/O_ref) < 1e-2

	# TO DO: test fout
	# TO DO: test functions of trajs_IO.jl
end
