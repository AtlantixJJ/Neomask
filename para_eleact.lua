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
    print("Parallel Model (Elementwise Bias and Scale Activation) Initialization.")
    ParallelNet.__init(self, config, layer)
end

function ParallelNet_Mod:make_NewActivation(activation)
    print("ParallelNet_EleAct Activation Building...")
    print("Using activation : ")
    print(activation())
    local input_layer = nn.ParallelTable()
    activation = activation or cudnn.Tanh
    input_layer:add(nn.Identity()) -- nn.CMul(self.from_nfeat, self.from_size, self.from_size) -- use multiple
    --input_layer:add(nn.Sequential():add(nn.Mul()):add(nn.Add(scalar=true)):add(activation)) -- use single scale factor
    input_layer:add(nn.Sequential():add(nn.CMul(self.from_nfeat, self.from_size, self.from_size)):add(nn.CAdd(self.from_nfeat, self.from_size, self.from_size)):add(activation()))
    self.new_act = nn.Sequential():add(nn.CMulTable(input_layer)):cuda()
end

return nn.ParallelNet_Mod