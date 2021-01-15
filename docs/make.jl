using Pkg, KnetMetrics;
Pkg.add("Documenter")
using Documenter

makedocs(
    modules = [KnetMetrics],
    format = Documenter.HTML(),
    sitename = "KnetMetrics.jl",
    authors = "Emirhan Kurtuluş.",
    pages = Any[
        "Home" => "index.md",
        "Manual" => Any[
            "tutorial.md",
#           "examples.md",
            "reference.md"
        ]])


deploydocs(repo = "https://github.com/emirhan422/KnetMetrics.jl.git", push_preview = true) 
