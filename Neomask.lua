require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'
local Neomask,_ = torch.class("nn.Neomask",'nn.Container')

function Neomask:__init(config)
    print("Neomask Model...")
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2

    cutorch.setDevice(self.gpu1)
    self.M = config.model:cuda()
    local tempin = torch.rand(12,3,224,224):cuda()
    local temp = self.M:forward(tempin)
    self.M:backward(tempin,temp)
    self.config = config
    self.name = config.name

    self.ks = 3
    self.pd = 1
    self.fs = 64

    self.lIns = {}
    self.lH = 10
    self.lL = 6

    cutorch.setDevice(self.gpu2)
    self:build_extra()
    self:build_tail()

end

function Neomask:build_tail()
    -- scale up to input    
    --print(self.M.modules[self.layer].gradInput:size())
    local scale = 2 -- 224 / self.M.modules[self.layer].gradInput:size(3)
    self.tail = nn.Sequential()

    self.tail:add(nn.SpatialUpSamplingBilinear(scale))
    self.tail:add(nn.View(self.config.batch,self.config.gSz*self.config.gSz))

    self.tail = self.tail:cuda()
end

function Neomask:build_Neuint(layer,scale)
    local Neuint = nn.Sequential()
    local forward_feat = nn.Sequential()
    local backward_feat = nn.Sequential()
    local fb_feat = nn.Sequential()

    local nFeat = self.M.modules[layer].gradInput:size(2)

    print(self.fs/scale)

    forward_feat:add(nn.SpatialZeroPadding(self.pd,self.pd))
    forward_feat:add(cudnn.SpatialConvolution(nFeat,self.fs/scale,self.ks,self.ks,1,1))
    forward_feat:add(nn.SpatialBatchNormalization(self.fs/scale))
    forward_feat:add(cudnn.ReLU(true))

    backward_feat:add(nn.SpatialZeroPadding(self.pd,self.pd))
    backward_feat:add(cudnn.SpatialConvolution(nFeat,self.fs/scale,self.ks,self.ks,1,1))
    backward_feat:add(nn.SpatialBatchNormalization(self.fs/scale))
    backward_feat:add(cudnn.ReLU(true))

    fb_feat:add(nn.ParallelTable():add(forward_feat):add(backward_feat))
    fb_feat:add(nn.JoinTable(2)) -- join at (batchNeuint

    fb_feat:add(nn.SpatialZeroPadding(self.pd,self.pd))
    fb_feat:add(cudnn.SpatialConvolution(2*self.fs/scale,self.fs/scale,self.ks,self.ks,1,1))
    fb_feat:add(nn.SpatialBatchNormalization(self.fs/scale))
    -- Neuint:add(cudnn.ReLU(true))

    Neuint:add(nn.JoinTable(2))
    Neuint:add(nn.SpatialZeroPadding(self.pd,self.pd))
    Neuint:add(cudnn.SpatialConvolution(2*self.fs/scale,self.fs/scale/2,self.ks,self.ks,1,1))
    Neuint:add(nn.SpatialBatchNormalization(self.fs/scale/2))    

    return Neuint:cuda()
end

function Neomask:build_extra()
    local scale = 1
    self.neuints = {}
    for i=self.lL,self.lH do
        self.neuints[i] = self:build_Neuint(i,scale)
        scale = scale * 2
    end
end

function Neomask:test_forward(input_batch,layer)
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

function Neomask:forward(input_batch)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    local pre = self.M:forward(self.inputs) * 100
    local N = #self.M.modules

    self.M:zeroGradParameters()
    for i=1,self.lL do 
        pre = self.M.modules[N-i+1]:backward(self.M.modules[N-i].output, pre)
    end

    -- Build refine input
    cutorch.setDevice(self.gpu2)
    for i=self.lL,self.lH do
        self.lIns[i] = {self.M.modules[N-i].output:clone(), self.M.modules[N-i+1].gradInput:clone()}
    end

    for i=self.lL,self.lH do
        self.lIns[i][3] = pre
        --print(self.lIns[i])
        pre = self.neuints[i]:forward(self.lIns[i])
    end

    print(pre:size())

    -- build net output
    -- self.tailin = self.gradFit:forward(pre) -- :clone()
    -- self.output = self.tail:forward(self.tailin) -- :clone()
    -- print(self.output:size())
    -- return self.output
end

function Neomask:backward(input_batch,gradInput)
    cutorch.setDevice(self.gpu2)
    self.tailgrad = self.tail:backward(self.tailin, gradInput) -- :clone()
    pre = self.tailgrad
    for i=self.lH,self.lL,-1 do
        pre = self.neuints[i]:backward(self.lIns[i],pre)
    end
    -- fdself.gradResin = self.M:backward(input_batch,self.gradFitin)
end

function Neomask:training()
    cutorch.setDevice(self.gpu1)
    self.M:training()
    cutorch.setDevice(self.gpu2)
    for k,n in pairs(self.neuints) do self.neuints[k]:training() end
    self.tail:training()
end

function Neomask:evaluate()
    cutorch.setDevice(self.gpu1)
    self.M:evaluate()
    cutorch.setDevice(self.gpu2)
    for k,n in pairs(self.neuints) do self.neuints[k]:evaluate() end
    self.tail:evaluate()
end

function Neomask:zeroGradParameters()
    cutorch.setDevice(self.gpu1)
    self.M:zeroGradParameters()
    cutorch.setDevice(self.gpu2)
    for k,n in pairs(self.neuints) do self.neuints[k]:zeroGradParameters() end
    self.tail:zeroGradParameters()
end

function Neomask:updateParameters(lr)
    -- self.M:updateParameters(lr)
    cutorch.setDevice(self.gpu2)
    for k,n in pairs(self.neuints) do self.neuints[k]:updateParameters(lr) end
    self.tail:updateParameters(lr)
end

function Neomask:cuda()
    cutorch.setDevice(self.gpu1)
    self.M:cuda()
    cutorch.setDevice(self.gpu2)
    for k,n in pairs(self.neuints) do self.neuints[k]:cuda() end
    self.tail:cuda()
end

function Neomask:float()
    self.M:float()
    for k,n in pairs(self.neuints) do self.neuints[k]:float() end
    self.tail:float()
end

function Neomask:clone(...)
    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self); f:seek(1)
    local clone = f:readObject(); f:close()

    if select('#',...) > 0 then
        clone.M:share(self.M,...)
        for k,n in pairs(self.neuints) do clone.neuints[k]:share(self.neuints[k],...) end
        clone.tail:share(self.tail,...)
    end

    return clone
end

return nn.Neomask
