"""
    kets_to_dm(kets; parallel_type, traceout)

Parallely computes a density matrix from an array of kets denoted `kets`.
If an array of indices is provided through the key argument `traceout`, then the
reduced density matrix is reconstructed only on the complementary subspace.

# Arguments
* `kets`: 1-dimensional array of kets.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads`, `:split_threads`, `:parfor` and
`:pmap`.
* `traceout`: Indices of some subspace to be traced out.

See also: [`kets_to_obs`](@ref)
"""
function kets_to_dm(kets::Array{T,1}; parallel_type::Symbol = :none,
        traceout::Union{Vector{Integer},Missing}=missing) where {T<:StateVector}
    @assert all([ket.basis == kets[1].basis for ket in kets]) "All kets must share a common basis"

    if parallel_type == :none
        𝒫(x) = ismissing(traceout) ? dm(x) : ptrace(x, traceout);
        ρ = ismissing(traceout) ? DenseOperator(first(kets).basis) : DenseOperator(𝒫(first(kets).basis));
        for i in 1:length(kets)
            @inbounds ρ.data .+= 𝒫(kets[i]).data;
        end
        return ρ / length(kets);
    elseif parallel_type == :threads
        return dm_threads(kets; traceout=traceout);
    elseif parallel_type == :pmap
        return dm_pmap(kets; traceout=traceout);
    elseif parallel_type == :parfor
        return dm_parfor(kets; traceout=traceout);
    elseif parallel_type == :split_threads
        return dm_split_threads(kets; traceout=traceout);
    else
        valptypes = [:none, :threads, :pmap, :parfor, :split_threads];
        error("Invalid parallel type. Type :$parallel_type not available.\n"*
              "Available types are: "*reduce(*,[":$t " for t in valptypes]))
    end
end;

function dm_threads(kets::Array{T,1}; traceout::Union{Vector{Integer},Missing}=missing) where {T<:StateVector}
    N = length(kets);
    𝒫(x) = ismissing(traceout) ? dm(x) : ptrace(x, traceout);
    b = ismissing(traceout) ? first(kets).basis : (𝒫(first(kets).basis)).basis_l;
    ρs = [DenseOperator(b) for i in 1:Threads.nthreads()];
    Threads.@threads for i in 1:length(kets)
        @inbounds ρs[Threads.threadid()].data .+= 𝒫(kets[i]).data;
    end
    @inbounds for i in 2:length(ρs)
        ρs[1].data .+= ρs[i].data;
    end
    return ρs[1]/N;
end

function dm_pmap(kets::Array{T,1}; traceout::Union{Vector{Integer},Missing}=missing) where {T<:StateVector}
    N = length(kets);
    𝒫(x) = ismissing(traceout) ? dm(x) : ptrace(x, traceout);
    wp = CachingPool(workers())
    r = RemoteChannel(1)
    b = ismissing(traceout) ? first(kets).basis : (𝒫(first(kets).basis)).basis_l;
    put!(r, b)
    @sync for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :ρ, DenseOperator(fetch(r)))))
    end
    pmap(wp,1:N;batch_size=cld(N,length(wp.workers))) do i
        @inbounds ρ.data .+= 𝒫(kets[i]).data;
    end
    R = DenseOperator(b);
    for w in wp.workers
        @inbounds R.data .+= @fetchfrom w ρ.data
    end
    @sync for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :ρ, nothing)))
    end
    return R/N;
end

function dm_parfor(kets::Array{T,1}; traceout::Union{Vector{Integer},Missing}=missing) where {T<:StateVector}
    N = length(kets);
    𝒫(x) = ismissing(traceout) ? dm(x) : ptrace(x, traceout);
    b = ismissing(traceout) ? first(kets).basis : (𝒫(first(kets).basis)).basis_l;
    wp = CachingPool(workers())
    r = RemoteChannel(1)
    put!(r, b)
    @sync for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :ρ, DenseOperator(fetch(r)))))
    end
    # @distributed (+) is here much slower
    @sync @distributed for i in 1:N
        @inbounds ρ.data .+= 𝒫(kets[i]).data;
    end
    R = DenseOperator(b);
    for w in wp.workers
        @inbounds R.data .+= @fetchfrom w ρ.data
    end
    @sync for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :ρ, nothing)))
    end
    return R/N;
end

function dm_split_threads(kets::Array{T,1}; traceout::Union{Vector{Integer},Missing}=missing) where {T<:StateVector}
    N = length(kets);
    𝒫(x) = ismissing(traceout) ? dm(x) : ptrace(x, traceout);
    wp = CachingPool(workers())
    r = RemoteChannel(1)
    b = ismissing(traceout) ? first(kets).basis : (𝒫(first(kets).basis)).basis_l;
    put!(r, b)
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :hb, fetch(r))));
    end
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, :(ρs = [DenseOperator(hb) for i in 1:Threads.nthreads()])));
    end
    batches = nfolds(1:length(kets),length(wp.workers))

    pmap(wp,1:length(wp.workers)) do i
        Threads.@threads for j in 1:length(batches[i])
            @inbounds ρs[Threads.threadid()].data .+= dm(kets[batches[i][j]]).data;
        end
        @inbounds for i in 2:length(ρs)
            ρs[1].data .+= ρs[i].data;
        end
    end
    R = DenseOperator(b);
    for w in wp.workers
        @inbounds R.data .+= @fetchfrom w ρs[1].data
    end
    @sync for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :ρs, nothing)))
    end
    return R/N;
end

# Surprisingly, slightly faster
function dm_split_threads_v2(kets::Array{T,1}; traceout::Union{Vector{Integer},Missing}=missing) where {T<:StateVector}
    N = length(kets);
    ρ = ismissing(traceout) ? DenseOperator(first(kets).basis) : DenseOperator(𝒫(first(kets).basis));
    𝒫(x) = ismissing(traceout) ? dm(x) : ptrace(x, traceout);
    wp = CachingPool(workers());
    remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
    acc = @async begin
        @inbounds for i in 1:Threads.nthreads()*length(wp.workers)
            ρ.data .+= take!(remch).data;
        end
        nothing
    end
    batches = nfolds(1:length(kets),length(wp.workers))
    pmap(wp,1:length(wp.workers)) do i
        ρs = ismissing(traceout) ? [DenseOperator(first(kets).basis) for i in 1:Threads.nthreads()] : [DenseOperator(𝒫(first(kets).basis)) for i in 1:Threads.nthreads()];
        Threads.@threads for ket in kets[batches[i]]
            @inbounds ρs[Threads.threadid()].data .+= 𝒫(ket).data;
        end
        for j in 1:length(ρs)
            put!(remch,ρs[j])
        end
    end
    fetch(acc);
    clear!(wp);
    return ρ/N;
end

"""
    kets_to_obs(op, kets; parallel_type, index)

Computes the quantum average of an operator over an array of kets
denoted `kets`. If an index is provided through the key argument `index`, then
the operator is only evaluated on the corresponding subspace.

# Arguments
* `op`: Arbitrary Operator.
* `parallel_type=:none`: The type of parallelism to employ. The types of
parallelism included are: `:none`, `:threads`, `:split_threads`, `:parfor` and
`:pmap`. In practice, use only `:threads` or `:none`.
* `index`: Indices of the subspace where one wants to evaluate `op`.

See also: [`kets_to_dm`](@ref)
"""
function kets_to_obs(op::AbstractOperator, kets::Array{T,1}; parallel_type::Symbol = :none,
        index=missing) where {T<:StateVector}
    if parallel_type == :none
        obs = zeros(ComplexF64,1);
        for i in 1:length(kets)
            @inbounds obs += ismissing(index) ? expect(op,kets[i]) : expect(index,op,kets[i]);
        end
        return obs / length(kets);
    elseif parallel_type == :threads
        expect_threads(op, kets; index=missing);
    elseif parallel_type == :pmap
        expect_pmap(op, kets; index=missing)
    elseif parallel_type == :parfor
        expect_parfor(op, kets; index=missing)
    elseif parallel_type == :split_threads
        expect_split_threads(op, kets; index=missing)
    else
        valptypes = [:none, :threads, :pmap, :parfor, :split_threads];
        error("Invalid parallel type. Type :$parallel_type not available.\n"*
              "Available types are: "*reduce(*,[":$t " for t in valptypes]))
    end
end;

function expect_threads(op::AbstractOperator, kets::Array{T,1}; index=missing) where {T<:StateVector}
    obss = zeros(ComplexF64,Threads.nthreads());
    Threads.@threads for i in 1:length(kets)
        @inbounds obss[Threads.threadid()] += ismissing(index) ? expect(op,kets[i]) : expect(index,op,kets[i]);
    end
    for i in 2:Threads.nthreads()
        @inbounds obss[1] += obss[i];
    end
    return obss[1] / length(kets);
end

function expect_pmap(op::AbstractOperator, kets::Array{T,1}; index=missing) where {T<:StateVector}
    N = length(kets);
    wp = CachingPool(workers())
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :obs_p, zeros(ComplexF64,1))));
    end
    pmap(wp,1:N;batch_size=cld(N,length(wp.workers))) do i
        @inbounds obs_p[] += ismissing(index) ? expect(op,kets[i]) : expect(index,op,kets[i]);
    end
    obs = zero(ComplexF64);
    @inbounds for w in wp.workers
        obs += @fetchfrom w obs_p[];
    end
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :obs_p, nothing)))
    end
    return obs / N;
end

function expect_parfor(op::AbstractOperator, kets::Array{T,1}; index=missing) where {T<:StateVector}
    obs = @distributed (+) for i in 1:length(kets)
        @inbounds ismissing(index) ? expect(op,kets[i]) : expect(index,op,kets[i]);
    end
    return obs / length(kets);
end

function expect_parfor_v2(op::AbstractOperator, kets::Array{T,1}; index=missing) where {T<:StateVector}
    N = length(kets);
    wp = CachingPool(workers())
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :obs_p, zeros(ComplexF64,1))));
    end
    @sync @distributed for i in 1:N
        @inbounds obs_p[] += ismissing(index) ? expect(op,kets[i]) : expect(index,op,kets[i]);
    end
    obs = zero(ComplexF64);
    @inbounds for w in wp.workers
        obs += @fetchfrom w obs_p[];
    end
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :obs_p, nothing)))
    end
    return obs / N;
end

function expect_split_threads(op::AbstractOperator, kets::Array{T,1}; index=missing) where {T<:StateVector}
    N = length(kets);
    wp = CachingPool(workers())
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, :(obss = zeros(ComplexF64,Threads.nthreads()))));
    end
    batches = nfolds(1:length(kets),length(wp.workers))
    pmap(wp,1:length(wp.workers)) do i
        Threads.@threads for j in 1:length(batches[i])
            @inbounds obss[Threads.threadid()] += ismissing(index) ? expect(op,kets[batches[i][j]]) : expect(index,op,kets[batches[i][j]]);
        end
        for i in 2:Threads.nthreads()
            @inbounds obss[1] += obss[i];
        end
    end
    obs = zero(ComplexF64);
    for w in wp.workers
        obs += @fetchfrom w obss[1];
    end
    @sync @async for w in wp.workers
        @spawnat(w, Core.eval(@__MODULE__, Expr(:(=), :obs_p, nothing)))
    end
    return obs / N;
end

# Slightly slower
function expect_split_threads_v2(op::AbstractOperator, kets::Array{T,1}; index=missing) where {T<:StateVector}
    N = length(kets);
    wp = CachingPool(workers());
    remch = RemoteChannel(()->Channel{Any}(Inf)); # TO DO: add some finite buffer size
    obs = zero(ComplexF64);
    acc = @async begin
        for i in 1:Threads.nthreads()*length(wp.workers)
            obs += take!(remch);
        end
        nothing
    end
    batches = nfolds(1:length(kets),length(wp.workers))
    pmap(wp,1:length(wp.workers)) do i
        obss = zeros(ComplexF64,Threads.nthreads());
        Threads.@threads for j in 1:length(batches[i])
            @inbounds obss[Threads.threadid()] += ismissing(index) ? expect(op,kets[batches[i][j]]) : expect(index,op,kets[batches[i][j]]);
        end
        for j in 1:length(obss)
            put!(remch,obss[j])
        end
    end
    fetch(acc);
    clear!(wp);
    return obs / N;
end
