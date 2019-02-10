"""
Module providing parallelised versions of [`QuantumOptics.timeevolution.mcwf`](@ref).
"""
module ParallelMCWF
export pmcwf

using Distributed, Base.Threads
using ProgressMeter, JLD2
import OrdinaryDiffEq
using QuantumOptics.bases, QuantumOptics.states, QuantumOptics.operators
using QuantumOptics.operators_dense, QuantumOptics.operators_sparse
using QuantumOptics.timeevolution
using QuantumOptics.operators_lazysum, QuantumOptics.operators_lazytensor, QuantumOptics.operators_lazyproduct
@everywhere using QuantumOptics.timeevolution.timeevolution_mcwf
const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Nothing}
Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)

"""
    pmcwf(tspan, psi0, H, J; Ntrajectories, <keyword arguments>)

Integrate parallely `Ntrajectories` MCWF trajectories.
Arguments are passed internally to `QuantumOptics.timeevolution.mcwf`.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
be displayed.
* `psi0`: Initial state vector.
* `H`: Arbitrary Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
operator type.
* `Ntrajectories=1`: Number of MCWF trajectories.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads` and `:pmap`.
* `progressbar=true`: If `true`, a progression bar is displayed.
* `return_data=true`: If `true`, the solution is returned as a `Tuple`.
* `save_data=true`: If `true`, the solution is saved to disk.
If `return_data=false`, less RAM is used (except for `parallel_type=:threads`).
* `fpath=missing`: savefile path (e.g. `some/valid/path/filename.jld2`).
Directory must pre-exist, the savefile is created.
* `additional_data=missing`: If given a `Dict`, entries are added to the
savefile.
* `seed=rand(UInt)`: Currently not supported except for `parallel_type=:none`.
* `rates=ones()`: Vector of decay rates.
* `fout`: If given, this function `fout(t, psi)` is called every time an
output should be displayed. ATTENTION: The state `psi` is neither
normalized nor permanent! It is still in use by the ode solve
and therefore must not be changed.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
operators. If they are not given they are calculated automatically.
* `display_beforeevent=false`: `fout` is called before every jump.
* `display_afterevent=false`: `fout` is called after every jump.
* `kwargs...`: Further arguments are passed on to the ode solver.

See also: [`timeevolution.mcwf`](@ref)

# Examples
```julia-repl
julia> tspan = Array(t0:dt:t_max);
julia> fb = FockBasis(10); ψ0 = fockstate(fb,0); a = destroy(fb);
julia> H = randoperator(fb); H = H + dagger(H); γ = 1.;
julia> # 100 MCWF trajectories
julia> t, trajs = pmcwf(Array(t0:dt:t_max), ψ0, H, [sqrt(γ)*a];
                        Ntrajectories=100, parallel_type=:pmap);
```
"""
function pmcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        Ntrajectories=1, parallel_type::Symbol = :none,
        progressbar::Bool = true,
        return_data::Bool = true, save_data::Bool = false,
        fpath::Union{String,Missing}=missing,
        additional_data::Union{Dict{String,T2},Missing}=missing,
        seed=rand(UInt), rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        kwargs...) where {B<:Basis,T<:Ket{B},T2}

    valptypes = [:none, :threads, :pmap];
    @assert parallel_type in valptypes "Invalid parallel type. Type :$parallel_type not available.\n"*
                                       "Available types are: "*reduce(*,[":$t " for t in valptypes])
    @assert return_data || save_data "pmcwf outputs nothing"
    save_data && @assert !ismissing(fpath) "ERROR: savefile path is missing"
    save_data && @assert isdir(splitdir(fpath)[1]) "ERROR: accessing "*splitdir(fpath)[1]*": No such directory"
    save_data && @assert !isfile(fpath) "ERROR: "*fpath*" already a file: Choose a free savefile name"

    if parallel_type == :none || Ntrajectories == 1
        return serial_mcwf(tspan,psi0,H,J;Ntrajectories=Ntrajectories,
            progressbar=progressbar,
            return_data=return_data,save_data=save_data,
            fpath=fpath,additional_data=additional_data,
            seed=seed,rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,
            kwargs...);
    elseif parallel_type == :threads
        # TO DO: seed argument not supported
        # TO DO: only save_data not totally supported. Could use less RAM by
        # writting trajectories directly to disk but would probably require
        # some locking or some parallel process.
        return multithreaded_mcwf(tspan,psi0,H,J;Ntrajectories=Ntrajectories,
            progressbar=progressbar,
            return_data=return_data,save_data=save_data,
            fpath=fpath,additional_data=additional_data,
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,
            kwargs...);
    elseif parallel_type == :pmap
        # TO DO: seed argument not supported
        # TO DO: add batch_size as an option
        return distributed_mcwf(tspan,psi0,H,J;Ntrajectories=Ntrajectories,
            progressbar=progressbar,
            return_data=return_data,save_data=save_data,
            fpath=fpath,additional_data=additional_data,
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,
            kwargs...);
    end
end

function serial_mcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        Ntrajectories=1, progressbar::Bool = true,
        return_data::Bool = true, save_data::Bool = false,
        fpath::Union{String,Missing}=missing,
        additional_data::Union{Dict{String,T2},Missing}=missing,
        seed=rand(UInt), rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        kwargs...) where {B<:Basis,T<:Ket{B},T2}
    if save_data
        file = jldopen(fpath, "a+");
        file["t"] = tspan;
        if !ismissing(additional_data)
            for (key, val) in additional_data
                file[key] = val;
            end
        end
    end
    if return_data
        # Pre-allocate an array for holding each MC simulation
        out_type = fout == nothing ? typeof(psi0) : pure_inference(fout, Tuple{eltype(tspan),typeof(psi0)});
        sols::Array{Vector{out_type},1} = fill(Vector{out_type}(),Ntrajectories);
    end
    if progressbar
        # A progress bar is set up to be updated by the master thread
        progress = Progress(Ntrajectories);
        ProgressMeter.update!(progress, 0);
    end
    for i in 1:Ntrajectories
        sol = timeevolution.mcwf(tspan,psi0,H,J;
            seed=seed,rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg,kwargs...);
        save_data ? file["trajs/"*string(i)] = sol[2] : nothing;
        return_data ? sols[i] = sol[2] : nothing;
        progressbar ? ProgressMeter.next!(progress) : nothing;
    end
    save_data && close(file);
    return return_data ? (tspan, sols) : nothing;
end

function multithreaded_mcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        Ntrajectories=1, progressbar::Bool = true,
        return_data::Bool = true, save_data::Bool = true,
        fpath::Union{String,Missing}=missing,
        additional_data::Union{Dict{String,T2},Missing}=missing,
        rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        kwargs...) where {B<:Basis,T<:Ket{B},T2}

    if progressbar
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
    end
    if save_data
        file = jldopen(fpath, "a+");
        file["t"] = tspan;
        if !ismissing(additional_data)
            for (key, val) in additional_data
                file[key] = val;
            end
        end
        return_data = true; # Some duct tape
    end
    if return_data
        # Pre-allocate an array for holding each MC simulation
        out_type = fout == nothing ? typeof(psi0) : pure_inference(fout, Tuple{eltype(tspan),typeof(psi0)});
        sols::Array{Vector{out_type},1} = fill(Vector{out_type}(),Ntrajectories);
    end
    # Multi-threaded for-loop over all MC trajectories.
    Threads.@threads for i in 1:Ntrajectories
        sol = timeevolution.mcwf(tspan,psi0,H,J;
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg, kwargs...);
        #save_data ? file["trajs/"*string(i)] = sol[2] : nothing;
        return_data ? sols[i] = sol[2] : nothing;
        # Updates progress bar if called from the main thread or adds a pending update otherwise
        progressbar ? update_progressbar(nupdates) : nothing;
    end
    progressbar ? update_progressbar(nupdates) : nothing;
    # Sets the progress bar to 100%
    if progressbar && (progress.counter < Ntrajectories) ProgressMeter.update!(progress, Ntrajectories); end;

    # Some additional duct tape
    if save_data
        for i in 1:length(sols)
            file["trajs/"*string(i)] = sols[i];
        end
    end

    save_data && close(file);
    return return_data ? (tspan, sols) : nothing;
end;

function distributed_mcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        Ntrajectories=1, progressbar::Bool = true,
        return_data::Bool = true, save_data::Bool = true,
        fpath::Union{String,Missing}=missing,
        additional_data::Union{Dict{String,T2},Missing}=missing,
        rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        alg=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        kwargs...) where {B<:Basis,T<:Ket{B},T2}

    # Create a remote channel from where trajectories are read out by the saver
    remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size

    wp = CachingPool(workers());

    # Create a task fetched by the first available worker that retrieves trajs
    # from the remote channel and writes them to disk. A progress bar is set up
    # as well.
    saver = @async launch_saver(remch; Ntrajectories=Ntrajectories,
        progressbar=progressbar, return_data=return_data, save_data=save_data,
        fpath=fpath, additional_data=additional_data);
    # Multi-processed for-loop over all MC trajectories. @async feeds workers()
    # with jobs from the local process and returns instantly. Jobs consist in
    # computing a trajectory and pipe it to the remote channel remch.
    tsk = @async pmap(wp, 1:Ntrajectories, batch_size=cld(Ntrajectories,length(wp.workers))) do i
        put!(remch, timeevolution.mcwf(tspan,psi0,H,J;
            rates=rates,fout=fout,Jdagger=Jdagger,
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            alg=alg, kwargs...));
        nothing
    end
    fetch(tsk);
    # Once saver has consumed all queued trajectories produced by all workers, an
    # array of MCWF trajs is returned.
    sols = fetch(saver);
    # Clear caching pool
    clear!(wp);

    if return_data
        out_type = fout == nothing ? typeof(psi0) : pure_inference(fout, Tuple{eltype(tspan),typeof(psi0)});
        return (tspan, convert(Array{Vector{out_type},1},sols));
    else
        nothing;
    end
end;

function launch_saver(readout_ch::RemoteChannel{Channel{T1}};
        Ntrajectories=1, progressbar::Bool = true, return_data::Bool = true,
        save_data::Bool = true, fpath::Union{String,Missing}=missing,
        additional_data::Union{Dict{String,T2},Missing}=missing) where {T1, T2}
    if progressbar
        # Set up a progress bar
        progress = Progress(Ntrajectories);
        ProgressMeter.update!(progress, 0);
    end
    if save_data
        file = jldopen(fpath, "a+");
        println("Saving data to ",fpath)
        if !ismissing(additional_data)
            for (key, val) in additional_data
                file[key] = val;
            end
        end
    end
    if return_data
        sols::Array{Any,1} = Array{Any,1}(undef,Ntrajectories);
    end

    # Current trajectory index
    currtraj::Int = 1;
    while currtraj <= Ntrajectories
        # Retrieve a queued traj
        sol = take!(readout_ch);
        if save_data && (currtraj == 1) file["t"] = sol[1]; end
        save_data ? file["trajs/" * string(currtraj)] = sol[2] : nothing;
        return_data ? sols[currtraj] = sol[2] : nothing;
        progressbar ? ProgressMeter.next!(progress) : nothing;
        currtraj += 1;
    end
    # Set progress bar to 100%
    if progressbar && (progress.counter < Ntrajectories) ProgressMeter.update!(progress, Ntrajectories); end;

    save_data && close(file);
    return return_data ? sols : nothing;
end;

end # module
