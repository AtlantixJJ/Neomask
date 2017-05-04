--[[
    Forward and backward combined.
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

paths.dofile('para_model.lua')

local ParallelNet_Mod,ParallelNet = torch.class("nn.ParallelNet_Mod",'nn.ParallelNet')

function ParallelNet_Mod:__init(config, layer)
    print("Parallel Model (Naive) Initialization.")
    ParallelNet.__init(self, config, layer)
end

return nn.ParallelNet_Mod