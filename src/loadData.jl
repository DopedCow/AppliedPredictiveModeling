using RData

export APLData

###############################################################################
#
# ALPData - Load the requested dataset into a dataframe
#
###############################################################################
"""
    APLData

Check that the dataset name is valid and if so load the data.

### Arguments

* `name` : a `::String` with the name of the dataset

### Returns

* `::DataFrame`

### Examples

"""
APLDataSets = ["Segmentation", "Solution"]

function APLData(name::String)

    if name ∈ APLDataSets
        if name == "Segmentation"
            rdata = RData.load("./data/segmentationOriginal.RData")
            df = rdata["segmentationOriginal"]
        elseif name == "Solution"
            # code missing
        end
        
    else
        print("The dataset name is not valid.")
    end
end
