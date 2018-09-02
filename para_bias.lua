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

function ParallelNet_Mod:change_Activation(activation)
    print("Build from scratch...")
    self:make_NewActivation(activation)
    print(self.new_act)
end

function ParallelNet_Mod:make_NewActivation(activation)
    local input_layer = nn.ParallelTable()
    local bias_branch = nn.Sequential()
    activation = activation or cudnn.Tanh
    print("Using activation : ")
    print(activation())
    input_layer:add(nn.Identity()) -- nn.CMul(self.from_nfeat, self.from_size, self.from_size) -- use multiple
    bias_branch:add(nn.Mul()):add(nn.Add(1,true)):add(cudnn.Tanh()) -- to scale
    bias_branch:add(nn.Mul()):add(nn.Add(1,true)) -- scale the result of Tanh
    input_layer:add(bias_branch) -- use single scale factor
    --input_layer:add(nn.Sequential():add(nn.CMul(self.from_nfeat, self.from_size, self.from_size)):add(nn.CAdd(self.from_nfeat, self.from_size, self.from_size)):add(activation))
    self.new_act = nn.Sequential():add(input_layer):add(nn.CAddTable()):cuda()
end

return nn.ParallelNet_Mod