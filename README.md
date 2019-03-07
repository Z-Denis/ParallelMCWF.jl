# ParallelMCWF.jl
Simple package providing parallelised versions of [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl)'s `timeevolution.mcwf` Monte-Carlo wave-function algorithm.

### Features
- Setup parallel Monte Carlo simulations based on `timeevolution.mcwf`. Multithreading (`Threads.@threads for`), distributed computing (both `pmap` and `@distributed for`) and multithreading on split processes is supported.
- Support progress bar. Only the default [ProgressMeter](https://github.com/timholy/ProgressMeter.jl) progress bar is currently supported.
- Support memory-efficient real-time saving of Monte Carlo trajectories.
- Build density matrices parallely from arrays of kets with support for both multithreading (`Threads.@threads`) and distributed computing (`pmap`).
- Average operators parallely over arrays of kets with support for both multithreading (`Threads.@threads`) and distributed computing (`pmap`).
- Save to and load from disk MCWF trajectories specifying the range for both the trajectories and the times.

# Examples

### `pmcwf`
```julia
using Distributed
# Get 10 processes
if nprocs() < 10 addprocs(10-nprocs()); end
using ParallelMCWF, QuantumOptics
tspan = collect(0:10);
fb = FockBasis(10); ψ0 = fockstate(fb,0); a = destroy(fb);
H = randoperator(fb); H = H + dagger(H); γ = 1.;
# 3000 MCWF trajectories
t, trajs = pmcwf(tspan, ψ0, H, [sqrt(γ)*a]; Ntrajectories=3000, progressbar=true,
		parallel_type=:pmap, return_data=true, save_data=true, fpath="/some/valid/path/filename.jld2");
```
```julia-repl
Saving data to /some/valid/path/filename.jld2
Progress: 100%|█████████████████████████████████████████| Time: 0:00:05
```

### `kets_to_dm`, `kets_to_obs`
```julia
ρ = kets_to_dm([trajs[i][end] for i in 1:length(trajs)];parallel_type=:pmap);
E = kets_to_obs(H,[trajs[i][end] for i in 1:length(trajs)];parallel_type=:pmap);
```

### `save_trajs`, `load_trajs`
```julia
params = Dict("H" => H, "J" => [sqrt(γ)*a])
save_trajs("/some/valid/path/filename2.jld2",(t,trajs); additional_data=params)
t, trajs2 = load_trajs("/some/valid/path/filename2.jld2")
jldopen("/some/valid/path/filename2.jld2",r) do file
	println(file["H"] == H)
end
```
```julia-repl
julia> true
```
