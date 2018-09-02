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
    print("Parallel Model (Combination) Initialization.")
    ParallelNet.__init(self, config, layer)
end

function ParallelNet_Mod:make_NewActivation()
    print("ParallelNet_Comb Activation Building...")
    self.new_act = nn.Sequential()
    local input_layer = nn.ParallelTable()
    local net_input = nn.Sequential()
    local net_rele = nn.Sequential()

    net_input:add(nn.SpatialSymmetricPadding(1,1,1,1))
    net_rele :add(nn.SpatialSymmetricPadding(1,1,1,1))
    net_input:add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat/2,3,3,1,1))
    net_rele :add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat/2,3,3,1,1))

    input_layer:add(net_input):add(net_rele)
    
    self.new_act:add(input_layer):add(nn.JoinTable(2)) -- join at (batch,x,256,256)
    self.new_act:add(nn.SpatialSymmetricPadding(1,1,1,1))
    self.new_act:add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat,3,3,1,1))

    self.new_act = self.new_act:cuda()
end

return nn.ParallelNet_Mod