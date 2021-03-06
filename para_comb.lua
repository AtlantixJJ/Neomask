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

function ParallelNet_Mod:change_Activation(new_activation)
    self.new_act.modules[#self.new_act] = new_activation():cuda()
    local para = self.new_act.modules[1]
    table.remove(para.modules[1].modules,#para.modules[1])
    table.insert(para.modules[2].modules,#para.modules[2]+1,new_activation():cuda())
    print(para)
    for i=1,2 do
        para.modules[i].modules[#para.modules[i]] = new_activation():cuda()
    end
    print("Now : ", self.new_act)
end

function ParallelNet_Mod:make_NewActivation(activation)
    print("ParallelNet_Comb Activation Building...")
    activation = activation or cudnn.Tanh
    print("Using activation : ")
    print(activation())
    self.new_act = nn.Sequential()
    local input_layer = nn.ParallelTable()
    local net_input = nn.Sequential()
    local net_rele = nn.Sequential()

    net_input:add(nn.SpatialSymmetricPadding(1,1,1,1))
    net_input:add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat/2,3,3,1,1))
    net_input:add(activation())

    net_rele :add(nn.SpatialSymmetricPadding(1,1,1,1))
    net_rele :add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat/2,3,3,1,1))
    net_rele :add(activation())

    input_layer:add(net_input):add(net_rele)
    
    self.new_act:add(input_layer):add(nn.JoinTable(2)) -- join at (batch,x,256,256)
    self.new_act:add(nn.SpatialSymmetricPadding(1,1,1,1))
    self.new_act:add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat,3,3,1,1))
    self.new_act:add(activation())
    self.new_act = self.new_act:cuda()
end

return nn.ParallelNet_Mod