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

function ParallelNet_Mod:__init(config, tb)
    print("Parallel Model (FCN-Naive) Initialization.")
    ParallelNet.__init(self, config, tb)
end

function ParallelNet_Mod:from_old()
    self:build_maskBranch()
    self:make_NewActivation(cudnn.Sigmoid)
end

function ParallelNet:make_NewActivation(activation)
    print("Make New Activation 1*1Conv")
    local input_layer = nn.ParallelTable()
    local branch = nn.Sequential()
    local nfeat = self.nfeat -- (1024,14,14)
    activation = activation or cudnn.Sigmoid
    print("Using activation : ")
    print(activation())
    input_layer:add(nn.Identity()) -- nn.CMul(self.from_size[1], self.from_size, self.from_size) -- use multiple
    branch:add(cudnn.SpatialFullConvolution(nfeat,nfeat,1,1,1,1,0,0,0,0))
    branch:add(activation())
    input_layer:add(branch) -- use single scale factor
    --input_layer:add(nn.Sequential():add(nn.CMul(self.from_size[1], self.from_size, self.from_size)):add(nn.CAdd(self.from_size[1], self.from_size, self.from_size)):add(activation))
    self.new_act = nn.Sequential():add(input_layer):add(nn.CMulTable()):cuda()
    print(self.new_act)
end

function ParallelNet:rele_NewActivation(activation)
    print("Make New Activation 1*1Conv")
    local input_layer = nn.ParallelTable()
    local trunk_out = nn.Sequential():add(nn.SelectTable(1)) -- select original
    local grad_out = nn.Sequential():add(nn.SelectTable(2))
    local rele_out = nn.CMulTable():add(trunk_out):add(grad_out)

    local branch = nn.Sequential()
    local nfeat = self.nfeat -- (1024,14,14)
    activation = activation or cudnn.Sigmoid
    print("Using activation : ")
    print(activation())
    input_layer:add(nn.Identity()) -- nn.CMul(self.from_size[1], self.from_size, self.from_size) -- use multiple
    branch:add(rele_out):add(cudnn.SpatialFullConvolution(nfeat,nfeat,1,1,1,1,0,0,0,0))
    branch:add(activation())
    input_layer:add(branch) -- use single scale factor
    --input_layer:add(nn.Sequential():add(nn.CMul(self.from_size[1], self.from_size, self.from_size)):add(nn.CAdd(self.from_size[1], self.from_size, self.from_size)):add(activation))
    self.new_act = nn.Sequential():add(input_layer):add(nn.CMulTable()):cuda()
    print(self.new_act)
end


return nn.ParallelNet_Mod