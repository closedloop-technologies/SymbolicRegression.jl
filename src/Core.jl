module CoreModule

include("Utils.jl")
include("ProgramConstants.jl")
include("Dataset.jl")
include("OptionsStruct.jl")
include("Operators.jl")
include("Options.jl")

import .ProgramConstantsModule:
    MAX_DEGREE, BATCH_DIM, FEATURE_DIM, RecordType, DATA_TYPE, LOSS_TYPE
import .DatasetModule: Dataset
import .OptionsStructModule: Options, ComplexityMapping, MutationWeights, sample_mutation
import .OptionsModule: Options
import .OperatorsModule:
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    div,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma,
    erf,
    erfc,
    atanh_clip

end
