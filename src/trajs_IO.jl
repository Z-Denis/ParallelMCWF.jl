"""
    load_trajs(fpath; trajrange=missing, timerange=missing)

Loads trajectories of a MCWF simulation from a .jld file at the provided `fpath`.

# Arguments
* `fpath`: savefile path (e.g. `some/valid/path/filename.jld2`). Directory and
file must exist.
* `trajrange`: MCWF trajectories to be loaded. Defaults to `1:Ntrajectories`.
* `timerange`: Timerange to be loaded. Defaults to the full timerange.

See also: [`save_trajs`](@ref)
"""
function load_trajs(fpath::String; trajrange::Union{UnitRange{T},Missing}=missing,
        timerange::Union{UnitRange{T},Missing}=missing) where T<:Integer
    @assert isfile(fpath) "ERROR: accessing "*fpath*": No such file or directory"
    times = nothing
    sols = nothing;
    jldopen(fpath, "r") do file
        fulltrajrange = 1:length(keys(file["trajs"]));
        if ismissing(trajrange)
            trajrange = fulltrajrange;
        else
            @assert trajrange[end] <= fulltrajrange[end] "The 'trajrange' argument limits must not overpass those of the saved data"
        end
        fulltimerange = 1:length(file["t"]);
        if ismissing(timerange)
            timerange = fulltimerange;
        else
            @assert timerange[end] <= fulltimerange[end] "The 'timerange' argument limits must not overpass those of the saved data"
        end
        times = file["t"][timerange]
        sols = [file["trajs/" * string(i)][t] for t in timerange, i in trajrange];
    end
    return times, sols
end;

"""
    split_last_integer(s)

For any input string "string_i", where `i` is an integer, returns ("string_", i).
Only used by [`save_trajs`](@ref)
"""
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
function safe_fpath(fpath::String)
    path = dirname(fpath);
    @assert isdir(path) "ERROR: accessing "*path*": No such directory"
	fpath, ext = splitext(fpath);
    while isfile(fpath*ext)
        nfpath::String, num::String = split_last_integer(fpath);
        nfpath *= num == "" ? "_1" : string(parse(Int, num)+1);
        fpath = nfpath;
    end
    return fpath*ext;
end;

"""
    save_trajs(fpath, res=missing; additional_data=missing)

Saves trajectories of a MCWF simulation to a .jld file at the provided `fpath`.

# Arguments
* `fpath`: savefile path (e.g. `some/valid/path/filename.jld2`).
Directory must pre-exist. If the file already exists, either an integer is
appended to the name or is incremented to prevent overwritting any file.
* `res=missing`: MCWF solutions on under the form `Tuple(times, sols)`.
* `additional_data=missing`: If given a `Dict`, entries are added to the
savefile.

See also: [`load_trajs`](@ref)
"""
function save_trajs(fpath::String, res::Union{Tuple{Vector{T1},Vector{T2}},Missing}=missing;
        additional_data::Union{Dict{String,T3},Missing}=missing) where {T1<:Real,T2,T3}
    if isdir(dirname(fpath))
		if isfile(fpath)
			nfpath = safe_fpath(fpath);
			@warn "$(basename(fpath)) already exists at $(dirname(fpath)).\nSaving to $nfpath..."
		end
        jldopen(nfpath, "a+") do file
            if !ismissing(res)
                times, trajs = res;
                file["t"] = times;
                for i in 1:length(trajs)
                    file["trajs/" * string(i)] = [trajs[i][t] for t in 1:length(times)];
                end
            end
            if !ismissing(additional_data)
                for (key, val) in additional_data
                    file[key] = val;
                end
            end
        end
    else
        path = dirname(fpath);
        fname = basename(fpath);
        printstyled("ERROR: accessing "*path*": No such file or directory\n",color=:light_red)
        println("Please type a valid path (like \"my/valid/path/\") or enter \"exit\"");
        npath = readline(stdin);
        if npath == "exit"
        elseif isdirpath(npath)
            save_trajs(npath*fname,res;additional_data=additional_data);
        else
            println("Entered new path not found. Create it? [y/n]");
            ans = readline(stdin);
            if ans == "y"
                mkdir(npath)
                save_trajs(npath*fname,res;additional_data=additional_data);
            end
        end
    end
end;
