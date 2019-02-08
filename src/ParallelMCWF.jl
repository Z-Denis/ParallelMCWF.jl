module ParallelMCWF
export multithreaded_mcwf, distributed_mcwf, read_saves

using Distributed, Base.Threads, ProgressMeter, JLD2
@everywhere using QuantumOptics, QuantumOptics.timeevolution.timeevolution_mcwf, DifferentialEquations
@everywhere using Random; Random.seed!(0)

# Operates the tensor product of all arguments
@everywhere ⨂(x) = length(x) > 1 ? reduce(⊗, x) : x[1];
# Operates the sum of all arguments
@everywhere ∑(x) = sum(x);

"""
Module providing parallelised versions of [`QuantumOptics.timeevolution.mcwf`](@ref).
"""
#module ParallelMCWF
#export multithreaded_mcwf, distributed_mcwf, tmap!

"""
    multithreaded_mcwf(tspan, psi0, H, J, nb_trajs; <keyword arguments>)

Integrate `nb_trajs` MCWF trajectories parallely on `Threads.nthreads()` threads.
Arguments are passed internally to `QuantumOptics.timeevolution.mcwf`.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
be displayed.
* `psi0`: Initial state vector. If set to `nothing`, trajectories are initialised
with kets uniformly drawn from the Hilbert space.
* `H`: Arbitrary Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
operator type.
* `nb_trajs=1`: Number of trajectories to be averaged.
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

See also: [`distributed_mcwf`](@ref), [`timeevolution.mcwf`](@ref)

# Examples
```julia-repl
julia> tspan = Array(t0:dt:t_max); nb_trajs = 100;
julia> fb = FockBasis(10); ψ0 = fockstate(fb,0); a = destroy(fb);
julia> H = randoperator(fb); H = H + dagger(H); γ = 1.;
julia> # 100 MCWF trajectories
julia> t, trajs = multithreaded_mcwf(collect(t0:dt:t_max), ψ0, H, [sqrt(γ)*a],nb_trajs);
```
"""
function multithreaded_mcwf(tspan, psi0, H, J::Vector, nb_trajs=1, dm_output=false;
        rates=nothing, fout=nothing, dmfout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()), kwargs...)
    # A progress bar is set up to be updated by the master thread
    progress = Progress(nb_trajs);
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

    # Set up random initialisation
    gbs = H.basis_l;
    #= old version with linear index
    # Create a linear index over all possible configurations
    #cartesian = CartesianIndices(Tuple([0:gbs.shape[i] for i in 1:length(gbs.shape)]));
    =#
    configs = collect(Tuple.(CartesianIndices(Tuple([0:gbs.shape[i]-1 for i in 1:length(gbs.shape)]))[:]));
    configs = [rand(configs) for i in 1:nb_trajs];

    config = configs[1]; # First configuration
    if typeof(gbs) <: CompositeBasis
        ψ0 = psi0 != nothing ? psi0 : ⨂([fockstate(gbs.bases[i],config[i]) for i in 1:length(gbs.shape)]);
    else
        ψ0 = psi0 != nothing ? psi0 : fockstate(gbs,config[1]);
    end
    times, sol = timeevolution.mcwf(tspan,ψ0,H,J;
        rates=rates,fout=fout,Jdagger=Jdagger,
        display_beforeevent=display_beforeevent,
        display_afterevent=display_afterevent,
        kwargs...);

    if dm_output # Compute the density matrix at each time step
        ρt = sol .⊗ dagger.(sol);
        DmType = typeof(ρt);
        # Pre-allocate an array for holding each thread's MC simulation
        ρts::Array{DmType,1} = fill(0.0im .* ρt,Threads.nthreads());

        # Multi-threaded for-loop over all MC trajectories. Trajectories are first
        # summed thread-wise and then all the thread-summed trajectories are summed
        # together and normalized.
        Threads.@threads for i in 2:nb_trajs
            config = configs[i]; # i-th configuration
            if typeof(gbs) <: CompositeBasis
                ψ0 = psi0 != nothing ? psi0 : ⨂([fockstate(gbs.bases[i],config[i]) for i in 1:length(gbs.shape)]);
            else
                ψ0 = psi0 != nothing ? psi0 : fockstate(gbs,config[1]);
            end
            times, ψt = timeevolution.mcwf(tspan,ψ0,H,J;
                rates=rates,fout=nothing,Jdagger=Jdagger,
                display_beforeevent=display_beforeevent,
                display_afterevent=display_afterevent,
                kwargs...);
            # Updates progress bar if called from the main thread or adds a pending update otherwise
            update_progressbar(nupdates);
            @inbounds ρts[Threads.threadid()] += ψt .⊗ dagger.(ψt); # The single traj solution is added to the thread's solution
        end
        # Sets the progress bar to 100%
        if progress.counter < nb_trajs ProgressMeter.update!(progress, nb_trajs); end;
        # Average over all thread's mean trajectories:
        for i in 1:Threads.nthreads()
            @inbounds ρt += ρts[i];
        end
        ρt /= nb_trajs; # Normalize

        return times, ρt;
    else # Compute all trajs' states
        # Pre-allocate an array for holding each thread's MC simulation
        sols::Array{typeof(sol),1} = fill(typeof(sol)(),nb_trajs);
        sols[1] = sol;
        # Multi-threaded for-loop over all MC trajectories.
        Threads.@threads for i in 2:nb_trajs
            config = configs[i]; # i-th configuration
            if typeof(gbs) <: CompositeBasis
                ψ0 = psi0 != nothing ? psi0 : ⨂([fockstate(gbs.bases[i],config[i]) for i in 1:length(gbs.shape)]);
            else
                ψ0 = psi0 != nothing ? psi0 : fockstate(gbs,config[1]);
            end
            times, sol = timeevolution.mcwf(tspan,ψ0,H,J;
                rates=rates,fout=fout,Jdagger=Jdagger,
                display_beforeevent=display_beforeevent,
                display_afterevent=display_afterevent,
                alg=alg, kwargs...);
            # Updates progress bar if called from the main thread or adds a pending update otherwise
            update_progressbar(nupdates);
            sols[i] = sol; # The single traj solution is added to the thread's solution
        end
        # Sets the progress bar to 100%
        if progress.counter < nb_trajs ProgressMeter.update!(progress, nb_trajs); end;

        if fout != nothing
            obs = [mean([sols[i][tt] for i in 1:nb_trajs]) for tt in 1:length(times)];
            return times, obs;
        elseif dmfout != nothing
            obs = Array{Any}(undef, length(times));
            tmap!(obs,1:length(times)) do tt
                dmfout(mean([sols[i][tt] ⊗ dagger(sols[i][tt]) for i in 1:nb_trajs]))
            end
            #obs = asyncmap((tt)->dmfout(mean([sols[i][tt] ⊗ dagger(sols[i][tt]) for i in 1:nb_trajs])), 1:length(times))
            #obs = [dmfout(mean([sols[i][tt] ⊗ dagger(sols[i][tt]) for i in 1:nb_trajs])) for tt in 1:length(times)];
            return times, obs;
        else
            return times, sols;
        end
    end
end;

"""
    distributed_mcwf(tspan, psi0, H, J, nb_trajs; <keyword arguments>)

Integrate `nb_trajs` MCWF trajectories parallely on `Distributed.nprocs()` processes.
Arguments are passed internally to `QuantumOptics.timeevolution.mcwf`.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
be displayed.
* `psi0`: Initial state vector. If set to `nothing`, trajectories are initialised
with kets uniformly drawn from the Hilbert space.
* `H`: Arbitrary Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
operator type.
* `nb_trajs=1`: Number of trajectories to be averaged.
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
julia> tspan = Array(t0:dt:t_max); nb_trajs = 100;
julia> fb = FockBasis(10); ψ0 = fockstate(fb,0); a = destroy(fb);
julia> H = randoperator(fb); H = H + dagger(H); γ = 1.;
julia> # 100 MCWF trajectories
julia> t, trajs = distributed_mcwf(Array(t0:dt:t_max), ψ0, H, [sqrt(γ)*a],nb_trajs);
```
"""
function distributed_mcwf(tspan, psi0, H, J::Vector, nb_trajs=1;
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
    configs = [rand(configs) for i in 1:nb_trajs];

    # Create a remote channel from where trajectories are read out by the saver
    remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size

    wp = CachingPool(workers());

    # Create a task fetched by the first available worker that retrieves trajs
    # from the remote channel and writes them to disk. A progress bar is set up
    # as well.
    saver = @async launch_saver(fpath, remch, nb_trajs; additional_data=additional_data);
    # Multi-processed for-loop over all MC trajectories. @async feeds workers()
    # with jobs from the local process and returns instantly. Jobs consist in
    # computing a trajectory and pipe it to the remote channel remch.
    tsk = @async pmap(wp, 1:nb_trajs, batch_size=cld(nb_trajs,length(wp.workers))) do i
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
        obs = [mean([sols[i][tt] for i in 1:nb_trajs]) for tt in 1:length(times)];
        return tspan, obs;
    elseif dmfout != nothing
        obs = Array{Any}(undef, length(times));
        tmap!(obs,1:length(times)) do tt
            dmfout(mean([sols[i][tt] ⊗ dagger(sols[i][tt]) for i in 1:nb_trajs]))
        end
        return tspan, obs;
    else
        return tspan, sols;
    end
end;

#function to be called by the main worker that saves trajectories to a file.
function launch_saver(fpath::String, readout_ch::RemoteChannel{Channel{T1}}, nb_trajs::Int;
                      additional_data::Union{Dict{String,T2},Missing}=missing) where {T1, T2}
    println("Saver launched")
    println("Saving data to ",fpath)
    # Set up a progress bar
    progress = Progress(nb_trajs);
    ProgressMeter.update!(progress, 0);
    sols::Array{Any,1} = Array{Any,1}(undef,nb_trajs);

    # Current trajectory index
    currtraj::Int = 1;
    # Open once the destination file for storing all trajectories
    jldopen(fpath, "a+") do file
        if ismissing(additional_data) == false
            for (key, val) in additional_data
                file[key] = val;
            end
        end
        while currtraj <= nb_trajs
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
        if progress.counter < nb_trajs ProgressMeter.update!(progress, nb_trajs); end;
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
