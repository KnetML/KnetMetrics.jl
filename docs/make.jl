using Documenter, KnetMetrics

makedocs(
    modules = [KnetMetrics],
    format = Documenter.HTML(),
    sitename = "KnetMetrics.jl",
    authors = "Emirhan Kurtuluş.",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Manual" => Any[
            "install.md",
            "tutorial.md",
#           "examples.md",
            "reference.md",
        ])


deploydocs("https://github.com/emirhan422/KnetMetrics.jl.git")
