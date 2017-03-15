require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'

local DecompNet,_ = torch.class("nn.DecompNet",'nn.Container')

function DecompNet:__init(config)
    self.M = config.model
    self.config = config
    self.name = config.name

    self.ks = 5
    self.pd = 2
    self.fs = 16

    self.layer = 6

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

    self.gradFit:add(nn.View(self.config.batch,256*60*60))
    self.gradFit:add(nn.Normalize(2))
    self.gradFit:add(nn.View(self.config.batch,256,60,60))

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(256,self.fs,self.ks,self.ks,1,1))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(self.fs,2*self.fs,self.ks,self.ks,1,1))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(2*self.fs,1,self.ks,self.ks,1,1))

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
    -- input_batch should be CudaTensor
    self.inputs = input_batch
    local pred = self.M:forward(input_batch)
    local conf,ind = torch.max(pred, 2) -- max decomposition
    self.M:zeroGradParameters()

    -- build decomp output
    --[[
    local dectar = torch.zeros(input_batch:size(1), 1000)
    for i=1,input_batch:size(1) do dectar[i][ind[i][1] ] = 1 end
    dectar = torch.cmul(dectar,pred:double())
    ]]--

    -- build backward output
    -- self.M:zeroGradParameters()
    -- self.onehot_decomp = self.M:backward(input_batch,dectar:cuda()) -- :clone()

    self.M:zeroGradParameters()
    self.full_decomp = self.M:backward(input_batch,pred*100)

    -- self.full_decomp = self.M.modules[2].gradInput
    -- self.ss_in = torch.cmul(self.full_decomp,self.inputs)

    -- Build refine input
    self.ss_in = self.M.modules[self.layer].gradInput

    -- build net output
    self.tailin = self.gradFit:forward(self.ss_in) -- :clone()
    self.output = self.tail:forward(self.tailin) -- :clone()

    return self.output
end

function DecompNet:backward(input_batch,gradInput)
    self.tailgrad = self.tail:backward(self.tailin, gradInput) -- :clone()
    self.gradFitin = self.gradFit:backward(self.ss_in,self.tailgrad) -- :clone()
    -- fdself.gradResin = self.M:backward(input_batch,self.gradFitin)
end

function DecompNet:training()
    self.M:training()
    self.gradFit:training()
    self.tail:training()
end

function DecompNet:evaluate()
    self.M:evaluate()
    self.gradFit:evaluate()
    self.tail:evaluate()
end

function DecompNet:zeroGradParameters()
    self.M:zeroGradParameters()
    self.gradFit:zeroGradParameters()
    self.tail:zeroGradParameters()
end

function DecompNet:updateParameters(lr)
    -- self.M:updateParameters(lr)
    self.gradFit:updateParameters(lr)
    self.tail:updateParameters(lr)
end

function DecompNet:cuda()
    self.M:cuda()
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
