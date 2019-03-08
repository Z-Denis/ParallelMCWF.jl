# ParallelMCWF.jl
Simple package providing parallelised versions of [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl)'s `timeevolution.mcwf` Monte-Carlo wave-function algorithm.

## Features
- Setup parallel Monte Carlo simulations based on `timeevolution.mcwf`. Multithreading (`Threads.@threads for`), distributed computing (both `pmap` and `@distributed for`) and multithreading on split processes are supported.
- Support for progress bar. Only the default [ProgressMeter](https://github.com/timholy/ProgressMeter.jl) progress bar is currently supported.
- Support for memory-efficient real-time saving of Monte Carlo trajectories.
- Build density matrices in parallel from arrays of kets with support for both multithreading (`Threads.@threads for` one single or split processes) and distributed computing (`pmap`, `@distributed for`).
- Average operators in parallel over arrays of kets with support for both multithreading (`Threads.@threads for` one single or split processes) and distributed computing (`pmap`, `@distributed for`).
- Save to and load from disk MCWF trajectories specifying the range for both the trajectories and the times.

## Monte-Carlo wave-function problem

In addition to the parameters accepted by [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl)'s `timeevolution.mcwf`, `pmcwf` takes the following supplementary keyword arguments:
* `Ntrajectories=1`: Number of MCWF trajectories.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads`, `:split_threads`, `:parfor` and `:pmap`. In practice, use `:threads` on a local machine and `:pmap` otherwise.
	* `:none` just loops over `Ntrajectories` in a conventional `for` loop.
	* `:threads` loops over `Ntrajectories` in a `Threads.@threads for` loop.
	* `:split_threads` makes `Distributed.nworkers()` batches of trajectories, each multithreaded on its one process.
	* `:parfor` asynchronously distributes `Ntrajectories` among the workers via a `@sync @distributed for` loop.
	* `:pmap` asynchronously distributes `Ntrajectories` among the workers via a `pmap`.
* `progressbar=true`: If `true`, a [ProgressMeter](https://github.com/timholy/ProgressMeter.jl) progress bar is displayed with default settings.
* `return_data=true`: If `true`, the solution is returned as a `Tuple`.
* `save_data=true`: If `true`, the solution is saved to disk. If `return_data=false`, less RAM is used (except for `parallel_type=:threads`).
* `fpath=missing`: savefile path (e.g. `some/valid/path/filename.jld2`).
Directory must pre-exist, the savefile is created.
* `additional_data=missing`: If given a `Dict`, entries are added to the
savefile.

Except for `parallel_type=:threads`, trajectories are computed in various processes each of which puts its finished MC trajectories into a remote channel from which they are retrieved and written to disk from the main process. The main process also carries out the update of the progress bar.

## Density matrix from an array of kets

The function `ket_to_dm` allows one to build a density matrix from kets in parallel. It takes the following arguments:
* `kets`: 1-dimensional array of kets that sample the desired density matrix.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads`, `:split_threads`, `:parfor` and `:pmap`. The most efficient option is `:threads` for small Hilbert space sizes and `:pmap` when the basis' dimension is above a few thousand states.
* `traceout`: When the basis vectors are tensor products of some local basis, the reduced density matrix can efficiently computed on some desired subspace by specifying the indices to be traced out as an array.

## Observable average from an array of kets

* `op`: Some operator to be averaged.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads`, `:split_threads`, `:parfor` and `:pmap`. The most efficient option is always `:threads`. Other options perform very badly here.
* `index`: Indices of the subspace where one wants to evaluate `op` as passed to [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl)'s `expect` function.

## Examples

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
