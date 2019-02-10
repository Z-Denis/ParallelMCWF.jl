"""
Module providing parallelised versions of [`QuantumOptics.timeevolution.mcwf`](@ref).
"""
module ParallelMCWF
export pmcwf, multithreaded_mcwf, distributed_mcwf, read_saves

using Distributed, Base.Threads
using ProgressMeter, JLD2#, QuantumOptics
import OrdinaryDiffEq
using QuantumOptics.bases, QuantumOptics.states, QuantumOptics.operators
using QuantumOptics.operators_dense, QuantumOptics.operators_sparse
using QuantumOptics.timeevolution
using QuantumOptics.operators_lazysum, QuantumOptics.operators_lazytensor, QuantumOptics.operators_lazyproduct
@everywhere using QuantumOptics.timeevolution.timeevolution_mcwf#, QuantumOptics
const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Nothing}
Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)

# Operates the tensor product of all arguments
@everywhere ⨂(x) = length(x) > 1 ? reduce(⊗, x) : x[1];
# Operates the sum of all arguments
@everywhere ∑(x) = sum(x);

function pmcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        Ntrajectories=1, parallel_type::Symbol = :none,
        seed=rand(UInt), rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        kwargs...) where {B<:Basis,T<:Ket{B}}
    @assert parallel_type in [:none, :threads, :pmap] "Invalid parallel type. Type '$parallel_type' not available."

    if parallel_type == :none
        return serial_mcwf(tspan,psi0,H,J;Ntrajectories=Ntrajectories,
            seed=seed,rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,
            kwargs...);
    elseif parallel_type == :threads
        return multithreaded_mcwf(tspan,psi0,H,J;Ntrajectories=Ntrajectories,
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,
            kwargs...);
    elseif parallel_type == :pmap
        return distributed_mcwf(tspan,psi0,H,J;Ntrajectories=Ntrajectories,
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,
            kwargs...);
    end
end

function serial_mcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        Ntrajectories=1, seed=rand(UInt), rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        kwargs...) where {B<:Basis,T<:Ket{B}}
    # Pre-allocate an array for holding each MC simulation
    out_type = fout == nothing ? typeof(psi0) : pure_inference(fout, Tuple{eltype(tspan),typeof(psi0)});
    sols::Array{Tuple{Vector{Float64},Vector{out_type}},1} = fill((Vector{Float64}(),Vector{out_type}()),Ntrajectories);
    for i in 1:Ntrajectories
        sols[i] = timeevolution.mcwf(tspan,psi0,H,J;
            seed=seed, rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg, kwargs...);
    end
    return sols;
end

"""
    multithreaded_mcwf(tspan, psi0, H, J; Ntrajectories, <keyword arguments>)

Integrate `Ntrajectories` MCWF trajectories parallely on `Threads.nthreads()` threads.
Arguments are passed internally to `QuantumOptics.timeevolution.mcwf`.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
be displayed.
* `psi0`: Initial state vector. If set to `nothing`, trajectories are initialised
with kets uniformly drawn from the Hilbert space.
* `H`: Arbitrary Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
operator type.
* `Ntrajectories=1`: Number of trajectories to be averaged.
* `seed=rand()`: Seed used for the random number generator.
* `rates=ones()`: Vector of decay rates.
* `fout`: If given, this function `fout(t, psi)` is called every time an
output should be displayed.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
operators. If they are not given they are calculated automatically.
* `display_beforeevent=false`: `fout` is called before every jump.
* `display_afterevent=false`: `fout` is called after every jump.
* `kwargs...`: Further arguments are passed on to the ode solver.

See also: [`distributed_mcwf`](@ref), [`timeevolution.mcwf`](@ref)

# Examples
```julia-repl
julia> tspan = Array(t0:dt:t_max); Ntrajectories = 100;
julia> fb = FockBasis(10); ψ0 = fockstate(fb,0); a = destroy(fb);
julia> H = randoperator(fb); H = H + dagger(H); γ = 1.;
julia> # 100 MCWF trajectories
julia> t, trajs = multithreaded_mcwf(collect(t0:dt:t_max), ψ0, H, [sqrt(γ)*a],Ntrajectories);
```
"""
function multithreaded_mcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        Ntrajectories=1, rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        kwargs...) where {B<:Basis,T<:Ket{B}}

    if Ntrajectories == 1
        return timeevolution.mcwf(tspan,psi0,H,J;
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,
            kwargs...);
    end

    # A progress bar is set up to be updated by the master thread
    progress = Progress(Ntrajectories);
    ProgressMeter.update!(progress, 0);
    function update_progressbar(n::Threads.Atomic{Int64})
        if Threads.threadid() == 1  # If first thread: update progress bar
            for i in 1:(n[]+1) ProgressMeter.next!(progress); end
            Threads.atomic_xchg!(n,0);
        else                    # Else: increment the number of pending updates.
            Threads.atomic_add!(n,1);
        end
    end
    nupdates = Threads.Atomic{Int}(0);

    # Pre-allocate an array for holding each MC simulation
    out_type = fout == nothing ? typeof(psi0) : pure_inference(fout, Tuple{eltype(tspan),typeof(psi0)});
    sols::Array{Tuple{Vector{Float64},Vector{out_type}},1} = fill((Vector{Float64}(),Vector{out_type}()),Ntrajectories);
    # Multi-threaded for-loop over all MC trajectories.
    Threads.@threads for i in 1:Ntrajectories
        sols[i] = timeevolution.mcwf(tspan,psi0,H,J;
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg, kwargs...);
        #Core.println(sols[i][2][end].data)
        # Updates progress bar if called from the main thread or adds a pending update otherwise
        update_progressbar(nupdates);
    end
    # Sets the progress bar to 100%
    if progress.counter < Ntrajectories ProgressMeter.update!(progress, Ntrajectories); end;

    return sols;
end;

"""
    distributed_mcwf(tspan, psi0, H, J; Ntrajectories, <keyword arguments>)

Integrate `Ntrajectories` MCWF trajectories parallely on `Distributed.nprocs()` processes.
Arguments are passed internally to `QuantumOptics.timeevolution.mcwf`.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
be displayed.
* `psi0`: Initial state vector. If set to `nothing`, trajectories are initialised
with kets uniformly drawn from the Hilbert space.
* `H`: Arbitrary Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
operator type.
* `Ntrajectories=1`: Number of trajectories to be averaged.
* `rates=ones()`: Vector of decay rates.
* `fout`: If given, this function `fout(t, psi)` is called every time an
output should be displayed.
* `dmfout`: If given, this function `dmfout(rho)` is called every time an
output should be displayed.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
operators. If they are not given they are calculated automatically.
* `display_beforeevent=false`: `fout` is called before every jump.
* `display_afterevent=false`: `fout` is called after every jump.
* `kwargs...`: Further arguments are passed on to the ode solver.

See also: [`multithreaded_mcwf`](@ref), [`timeevolution.mcwf`](@ref)

# Examples
```julia-repl
julia> tspan = Array(t0:dt:t_max); Ntrajectories = 100;
julia> fb = FockBasis(10); ψ0 = fockstate(fb,0); a = destroy(fb);
julia> H = randoperator(fb); H = H + dagger(H); γ = 1.;
julia> # 100 MCWF trajectories
julia> t, trajs = distributed_mcwf(Array(t0:dt:t_max), ψ0, H, [sqrt(γ)*a],Ntrajectories);
```
"""
function distributed_mcwf(tspan, psi0, H, J::Vector; Ntrajectories=1,
        additional_data::Union{Dict{String,T},Missing}=missing,
        fpath::String="Persistent_current/Datafiles/"*safe_save_name("Persistent_current/Datafiles/", "data")*".jld2",
        rates=nothing, fout=nothing, dmfout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()), kwargs...) where T

    # Set up random initialisation
    gbs = H.basis_l;
    #= old version with linear index
    # Create a linear index over all possible configurations
    #cartesian = CartesianIndices(Tuple([0:gbs.shape[i] for i in 1:length(gbs.shape)]));
    =#
    configs = collect(Tuple.(CartesianIndices(Tuple([0:gbs.shape[i] for i in 1:length(gbs.shape)]))[:]));
    configs = [rand(configs) for i in 1:Ntrajectories];

    # Create a remote channel from where trajectories are read out by the saver
    remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size

    wp = CachingPool(workers());

    # Create a task fetched by the first available worker that retrieves trajs
    # from the remote channel and writes them to disk. A progress bar is set up
    # as well.
    saver = @async launch_saver(fpath, remch, Ntrajectories; additional_data=additional_data);
    # Multi-processed for-loop over all MC trajectories. @async feeds workers()
    # with jobs from the local process and returns instantly. Jobs consist in
    # computing a trajectory and pipe it to the remote channel remch.
    tsk = @async pmap(wp, 1:Ntrajectories, batch_size=cld(Ntrajectories,length(wp.workers))) do i
        config = configs[i]; # i-th configuration
        ψ0 = psi0 != nothing ? psi0 : ⨂([fockstate(ghs.bases[i],config[i]) for i in 1:length(gbs.shape)]);
        sol = timeevolution.mcwf(tspan,ψ0,H,J;
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg, kwargs...);
        put!(remch, sol);
        nothing
    end
    fetch(tsk);
    # Once saver has consumed all queued trajectories produced by all workers, an
    # array of MCWF trajs is returned.
    sols = fetch(saver);

    # Clear caching pool
    clear!(wp);

    # Further processing of the MCWF trajectories.
    if fout != nothing
        obs = [mean([sols[i][tt] for i in 1:Ntrajectories]) for tt in 1:length(times)];
        return tspan, obs;
    elseif dmfout != nothing
        obs = Array{Any}(undef, length(times));
        tmap!(obs,1:length(times)) do tt
            dmfout(mean([sols[i][tt] ⊗ dagger(sols[i][tt]) for i in 1:Ntrajectories]))
        end
        return tspan, obs;
    else
        return tspan, sols;
    end
end;

#function to be called by the main worker that saves trajectories to a file.
function launch_saver(fpath::String, readout_ch::RemoteChannel{Channel{T1}}, Ntrajectories::Int;
                      additional_data::Union{Dict{String,T2},Missing}=missing) where {T1, T2}
    println("Saver launched")
    println("Saving data to ",fpath)
    # Set up a progress bar
    progress = Progress(Ntrajectories);
    ProgressMeter.update!(progress, 0);
    sols::Array{Any,1} = Array{Any,1}(undef,Ntrajectories);

    # Current trajectory index
    currtraj::Int = 1;
    # Open once the destination file for storing all trajectories
    jldopen(fpath, "a+") do file
        if ismissing(additional_data) == false
            for (key, val) in additional_data
                file[key] = val;
            end
        end
        while currtraj <= Ntrajectories
            # Retrieve a queued traj
            times, sols[currtraj] = take!(readout_ch);
            if currtraj == 1 file["t"] = times; end
            # Save it to disk
            file["trajs/" * string(currtraj)] = sols[currtraj];
            # Update progress bar
            ProgressMeter.next!(progress);
            currtraj += 1;
        end
        # Set progress bar to 100%
        if progress.counter < Ntrajectories ProgressMeter.update!(progress, Ntrajectories); end;
    end
    return sols;
end;

function read_saves(path::String, fname::String; trajrange=nothing, timerange=nothing)
    times = nothing
    sols = nothing;
    jldopen(path * fname * ".jld2", "r") do file
        fulltrajrange = 1:length(keys(file["trajs"]));
        if trajrange == nothing
            trajrange = fulltrajrange;
        else
            @assert trajrange[end] <= fulltrajrange[end] "The trajrange argument limits must not overpass those of the saved data"
        end
        fulltimerange = 1:length(file["t"]);
        if timerange == nothing
            timerange = fulltimerange;
        else
            @assert timerange[end] <= fulltimerange[end] "The timerange argument limits must not overpass those of the saved data"
        end
        times = file["t"][timerange]
        sols = [file["trajs/" * string(i)][t] for t in timerange, i in trajrange];
    end
    return times, sols
end

function split_last_integer(s::String)
    num::String = ""
    for c::Char in s
        try
            n = parse(Int, c);
            num *= c;
        catch
            num="";
        end
    end
    return SubString(s,1,length(s)-length(num)), num;
end;
function safe_save_name(path::String, fname::String)
    @assert ispath(path) "ERROR: accessing "*path*": No such file or directory"
    while isfile(path * fname * ".jld2")
        nfname::String, num::String = split_last_integer(fname);
        nfname *= num == "" ? "_1" : string(parse(Int, num)+1);
        fname = nfname;
    end
    return fname;
end;

"""
    tmap!(function, destination, collection)
Multithreaded version of [`map!`](@ref).

See also: [`map!`](@ref)

# Examples
```julia-repl
julia> src = collect(1:15);
julia> dst = similar(src);
julia> @btime map!(dst,src) do x
           Libc.systemsleep(1)
           x
       end
  15.039 s (0 allocations: 0 bytes)
julia> @btime tmap!(dst,src) do x
           Libc.systemsleep(1)
           x
       end
  3.006 s (1 allocation: 32 bytes)
```
"""
function tmap!(f, dst, src)
    @assert length(dst) == length(src) "Source and destination containers must have same lengths"
    Threads.@threads for i in eachindex(src)
        dst[i] = f(src[i]);
    end
end;


end # module
