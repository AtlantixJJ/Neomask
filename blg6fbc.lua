require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'
local DecompNet,_ = torch.class("nn.DecompNet",'nn.Container')
function DecompNet:__init(config)
    print("BaseLine Model From Layer 6. Forward-Backward Combined.")
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2

    cutorch.setDevice(self.gpu1)
    self.M = config.model:cuda()
    self.config = config
    self.name = config.name

    self.ks = 3
    self.pd = 1
    self.fs = 64

    self.layer = 6

    cutorch.setDevice(self.gpu2)
    self:build_extra()
    self:build_tail()
end

function DecompNet:build_tail()
    -- scale up to input    
    --print(self.M.modules[self.layer].gradInput:size())
    local scale = 4 -- 224 / self.M.modules[self.layer].gradInput:size(3)
    self.tail = nn.Sequential()

    self.tail:add(nn.SpatialUpSamplingBilinear(scale))
    self.tail:add(nn.View(self.config.batch,self.config.gSz*self.config.gSz))

    self.tail = self.tail:cuda()
end

function DecompNet:build_extra()
    self.gradFit = nn.Sequential()
    local forward_feat = nn.Sequential()
    local backward_feat = nn.Sequential()

    print(self.fs)

    forward_feat:add(nn.SpatialZeroPadding(self.pd,self.pd))
    forward_feat:add(cudnn.SpatialConvolution(256,self.fs,self.ks,self.ks,1,1))
    forward_feat:add(nn.SpatialBatchNormalization(self.fs))
    forward_feat:add(cudnn.ReLU())

    forward_feat:add(nn.SpatialZeroPadding(self.pd,self.pd))
    forward_feat:add(cudnn.SpatialConvolution(self.fs,self.fs/2,self.ks,self.ks,1,1))
    forward_feat:add(nn.SpatialBatchNormalization(self.fs/2))
    forward_feat:add(cudnn.ReLU())

    backward_feat:add(nn.SpatialZeroPadding(self.pd,self.pd))
    backward_feat:add(cudnn.SpatialConvolution(256,self.fs,self.ks,self.ks,1,1))
    backward_feat:add(nn.SpatialBatchNormalization(self.fs))
    backward_feat:add(cudnn.ReLU())

    backward_feat:add(nn.SpatialZeroPadding(self.pd,self.pd))
    backward_feat:add(cudnn.SpatialConvolution(self.fs,self.fs/2,self.ks,self.ks,1,1))
    backward_feat:add(nn.SpatialBatchNormalization(self.fs/2))
    backward_feat:add(cudnn.ReLU())

    self.gradFit:add(nn.ParallelTable():add(forward_feat):add(backward_feat))
    self.gradFit:add(nn.JoinTable(2)) -- join at (batch,x,256,256)

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(self.fs,self.fs/2,self.ks,self.ks,1,1))
    self.gradFit:add(nn.SpatialBatchNormalization(self.fs/2))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(self.fs/2,self.fs/4,self.ks,self.ks,1,1))
    self.gradFit:add(nn.SpatialBatchNormalization(self.fs/4))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(self.fs/4,self.fs/8,self.ks,self.ks,1,1))
    self.gradFit:add(nn.SpatialBatchNormalization(self.fs/8))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(self.fs/8,1,self.ks,self.ks,1,1))

    self.gradFit = self.gradFit:cuda()
end

function DecompNet:test_forward(input_batch,layer)
    -- input_batch should be CudaTensor
    self.inputs = input_batch
    local pred = self.M:forward(input_batch)
    local conf,ind = torch.max(pred, 2) -- max decomposition

    self.M:zeroGradParameters()
    self.M:backward(input_batch,pred*100)

    local scale = 224 / self.M.modules[layer].gradInput:size(3)

    self.open_net = nn.Sequential()
    self.open_net:add(nn.Sum(2))
    self.open_net:add(nn.SpatialUpSamplingBilinear(scale))
    self.open_net:add(nn.SpatialZeroPadding(1,1))
    self.open_net:add(nn.SpatialAveragePooling(3,3,1,1))
    self.open_net = self.open_net:cuda()

    self.ss_in = self.open_net:forward(self.M.modules[layer].gradInput:cuda()):clone()
    self.ss_in1 = self.open_net:forward(self.M.modules[layer-1].output:cuda()):clone()
    
    -- print(torch.norm(self.M.modules[layer].gradInput,2),torch.norm(self.M.modules[layer-1].output,2))
    
    temp = torch.cmul(self.M.modules[layer].gradInput:clone(),self.M.modules[layer-1].output:clone())
    self.ss_in2 = self.open_net:forward(temp:cuda()):clone()
    -- print(self.M.modules[layer].gradInput:size(),self.ss_in:size(),self.M.modules[layer-1].output:size(),self.ss_in1:size())

    return {self.ss_in,self.ss_in1,self.ss_in2}
end

function DecompNet:forward(input_batch)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    local pre = self.M:forward(self.inputs)
    local N = #self.M.modules
    pre = pre * 100
    self.M:zeroGradParameters()
    for i=1,7 do
        pre = self.M.modules[N-i+1]:backward(self.M.modules[N-i].output, pre)
    end

    -- Build refine input
    cutorch.setDevice(self.gpu2)
    self.ss_in = {self.M.modules[self.layer-1].output:clone(),self.M.modules[self.layer].gradInput:clone()}
    -- print(self.ss_in)

    -- build net output
    self.tailin = self.gradFit:forward(self.ss_in) -- :clone()
    self.output = self.tail:forward(self.tailin) -- :clone()

    return self.output
end

function DecompNet:backward(input_batch,gradInput)
    cutorch.setDevice(self.gpu2)
    self.tailgrad = self.tail:backward(self.tailin, gradInput) -- :clone()
    self.gradFitin = self.gradFit:backward(self.ss_in,self.tailgrad) -- :clone()
    -- fdself.gradResin = self.M:backward(input_batch,self.gradFitin)
end

function DecompNet:training()
    cutorch.setDevice(self.gpu1)
    self.M:training()
    cutorch.setDevice(self.gpu2)
    self.gradFit:training()
    self.tail:training()
end

function DecompNet:evaluate()
    cutorch.setDevice(self.gpu1)
    self.M:evaluate()
    cutorch.setDevice(self.gpu2)
    self.gradFit:evaluate()
    self.tail:evaluate()
end

function DecompNet:zeroGradParameters()
    cutorch.setDevice(self.gpu1)
    self.M:zeroGradParameters()
    cutorch.setDevice(self.gpu2)
    self.gradFit:zeroGradParameters()
    self.tail:zeroGradParameters()
end

function DecompNet:updateParameters(lr)
    -- self.M:updateParameters(lr)
    cutorch.setDevice(self.gpu2)
    self.gradFit:updateParameters(lr)
    self.tail:updateParameters(lr)
end

function DecompNet:getParameters()
    return self.gradFit:getParameters()
end

function DecompNet:cuda()
    cutorch.setDevice(self.gpu1)
    self.M:cuda()
    cutorch.setDevice(self.gpu2)
    self.gradFit:cuda()
    self.tail:cuda()
end

function DecompNet:float()
    self.M:float()
    self.gradFit:float()
    self.tail:float()
end

function DecompNet:clone(...)
    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self); f:seek(1)
    local clone = f:readObject(); f:close()

    if select('#',...) > 0 then
        clone.M:share(self.M,...)
        clone.gradFit:share(self.gradFit,...)
        clone.tail:share(self.tail,...)
    end

    return clone
end

return nn.DecompNet
