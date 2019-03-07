"""
    kets_to_dm(kets; parallel_type, traceout)

Parallely computes a density matrix from an array of kets denoted `kets`.
If an array of indices is provided through the key argument `traceout`, then the
reduced density matrix is reconstructed only on the complementary subspace.

# Arguments
* `kets`: 1-dimensional array of kets.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads` and `:pmap`.
* `traceout`: Indices of some subspace to be traced out.

See also: [`kets_to_obs`](@ref)
"""
function kets_to_dm(kets::Array{T,1}; parallel_type::Symbol = :none,
        traceout::Union{Vector{Integer},Missing}=missing) where {T<:StateVector}
    valptypes = [:none, :threads, :pmap, :parfor, :split_threads];
    @assert parallel_type in valptypes "Invalid parallel type. Type :$parallel_type not available.\n"*
                                       "Available types are: "*reduce(*,[":$t " for t in valptypes])
    @assert all([ket.basis == kets[1].basis for ket in kets]) "All kets must share a common basis"
    # `traceout` is defined on all workers for `𝒫` to be properly defined everywhere
    r = RemoteChannel(myid())
    @spawnat(myid(), put!(r, traceout))
    @sync for w in workers()
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :traceout, fetch(r))))
    end
    𝒫(x) = ismissing(traceout) ? dm(x) : ptrace(x, traceout);
    ρ = ismissing(traceout) ? DenseOperator(first(kets).basis) : DenseOperator(𝒫(first(kets).basis));

    if parallel_type == :none
        for ket in kets
            @inbounds ρ .+= 𝒫(ket);
        end
    elseif parallel_type == :threads
        ρs = ismissing(traceout) ? [DenseOperator(first(kets).basis) for i in 1:Threads.nthreads()] : [DenseOperator(𝒫(first(kets).basis)) for i in 1:Threads.nthreads()];
        # Accumulate thread-wise
        Threads.@threads for ket in kets
            ρs[Threads.threadid()] .+= 𝒫(ket);
        end
        # Sum contributions from all threads
        for i in 1:Threads.nthreads()
            ρ .+= ρs[i];
        end
    elseif parallel_type == :pmap
        remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
        acc = @async begin
            for i in 1:length(kets)
                @inbounds ρ .+= take!(remch);
            end
            nothing
        end
        wp = CachingPool(workers());
        @sync @async pmap(wp, kets, batch_size=cld(length(kets),length(wp.workers))) do ket
            put!(remch, dm(ket));
            nothing
        end
        fetch(acc);
        clear!(wp);
    elseif parallel_type == :parfor
        remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
        wp = CachingPool(workers());
        acc = @async begin
            for i in 1:length(kets)
                @inbounds ρ .+= take!(remch);
            end
            nothing
        end
        @sync @distributed for ket in kets
            put!(remch, 𝒫(ket));
        end
        fetch(acc);
        clear!(wp);
    elseif parallel_type == :split_threads
        wp = CachingPool(workers());
        remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
        acc = @async begin
            for i in 1:Threads.nthreads()*length(wp.workers)
                @inbounds ρ .+= take!(remch);
            end
            nothing
        end
        batches = nfolds(1:length(kets),length(wp.workers))
        @sync @async pmap(wp,1:length(wp.workers)) do i
            ρs = ismissing(traceout) ? [DenseOperator(first(kets).basis) for i in 1:Threads.nthreads()] : [DenseOperator(𝒫(first(kets).basis)) for i in 1:Threads.nthreads()];
            Threads.@threads for ket in kets[batches[i]]
                ρs[Threads.threadid()] .+= 𝒫(ket);
            end
            for ρ_thread in ρs
                put!(remch,ρ_thread)
            end
        end
        fetch(acc);
        clear!(wp);
    end
    return ρ / length(kets);
end;

"""
    kets_to_obs(op, kets; parallel_type, index)

Computes the quantum average of an operator over an array of kets
denoted `kets`. If an index is provided through the key argument `index`, then
the operator is only evaluated on the corresponding subspace.

# Arguments
* `op`: Arbitrary Operator.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads` and `:pmap`.
* `index`: Indices of the subspace where one wants to evaluate `op`.

See also: [`kets_to_dm`](@ref)
"""
function kets_to_obs(op::AbstractOperator, kets::Array{T,1}; parallel_type::Symbol = :none,
        index=missing) where {T<:StateVector}
    obs = zeros(ComplexF64,1);
    valptypes = [:none, :threads, :pmap, :parfor, :split_threads];
    @assert parallel_type in valptypes "Invalid parallel type. Type :$parallel_type not available.\n"*
                                       "Available types are: "*reduce(*,[":$t " for t in valptypes])
    if parallel_type == :none
        for ket in kets
            obs[] += ismissing(index) ? expect(op,ket) : expect(index,op,ket);
        end
    elseif parallel_type == :threads
        obss = zeros(ComplexF64,Threads.nthreads());
        Threads.@threads for ket in kets
            obss[Threads.threadid()] += ismissing(index) ? expect(op,ket) : expect(index,op,ket);
        end
        for i in 1:Threads.nthreads()
            obs[] += obss[i];
        end
    elseif parallel_type == :pmap
        remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
        wp = CachingPool(workers());
        acc = @async begin
            for i in 1:length(kets)
                obs[] += take!(remch);
            end
            nothing
        end
        @sync @async pmap(wp, kets, batch_size=cld(length(kets),length(wp.workers))) do ket
            put!(remch, ismissing(index) ? expect(op,ket) : expect(index,op,ket));
            nothing
        end
        fetch(acc);
        clear!(wp);
    elseif parallel_type == :parfor
        remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
        wp = CachingPool(workers());
        acc = @async begin
            for i in 1:length(kets)
                obs[] += take!(remch);
            end
            nothing
        end
        @sync @distributed for ket in kets
            put!(remch, ismissing(index) ? expect(op,ket) : expect(index,op,ket));
        end
        fetch(acc);
        clear!(wp);
    elseif parallel_type == :split_threads
        wp = CachingPool(workers());
        remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
        acc = @async begin
            for i in 1:Threads.nthreads()*length(wp.workers)
                obs[] += take!(remch);
            end
            nothing
        end
        batches = nfolds(1:length(kets),length(wp.workers))
        @sync @async pmap(wp,1:length(wp.workers)) do i
            obss = zeros(ComplexF64,Threads.nthreads());
            Threads.@threads for ket in kets[batches[i]]
                obss[Threads.threadid()] += ismissing(index) ? expect(op,ket) : expect(index,op,ket);
            end
            for obs_thread in obss
                put!(remch,obs_thread)
            end
        end
        fetch(acc);
        clear!(wp);
    end

    return obs[] / length(kets);
end;
