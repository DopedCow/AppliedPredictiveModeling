module APL

# list all the common packages below
using CSV
using DataFrames
using DataFramesMeta
using StatsBase
using VegaLite

# list all the other files to be included below
include("loadData.jl")

# list all the functionality that should be published below
# export

# define custom types


###############################################################################
#
# Various helper definitions
#
###############################################################################

"""
    helper(a, b)

Description of what the function does.

### Arguments
* `a` : a `Type` or `AnotherType`
* `b` : a `Type`

### Returns

* `::Type` or `::SomethingWeird`

### Examples

```jldoctest
julia> using APL

julia> helper(a = xyz, b = abc)

```
"""
function helper()
end

end