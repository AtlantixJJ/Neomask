require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

local DecompNet,_ = torch.class("nn.DecompNet",'nn.Container')

function DecompNet:__init(config,layer)
    print("Decomp BaseNet.")
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2

    cutorch.setDevice(self.gpu1)
    -- the preset model cannot be cuda model
    self.M = config.model:cuda()
    -- pre-running to determine shapes
    self:precheck()

    self.config = config
    self.name = config.name

    self.ks = 3
    self.pd = 1
    self.fs = 64

    self.layer = layer

    cutorch.setDevice(self.gpu2)
    self:build_extra()
    self:build_tail()
end

function DecompNet:precheck()
    local tempin = torch.rand(12,3,224,224):cuda()
    local temp = self.M:forward(tempin)
    self.M:backward(tempin,temp)
    self.in_size = self.M.modules[1].gradInput:size(3)
    self.from_size = self.M.modules[self.layer].gradInput:size(3)
    self.from_nfeat = self.M.modules[self.layer].gradInput:size(2)
    self.scale = self.in_size / self.from_size 
end

-- tail is used for clearing ups
function DecompNet:build_tail()
    -- scale up to input
    print("Scale %d : %d -> %d" % {scale, self.in_size, self.from_size})

    self.tail = nn.Sequential()

    self.tail:add(nn.SpatialUpSamplingBilinear(self.scale))
    -- need to convert to 1-d tensor to take in margin loss
    self.tail:add(nn.View(self.config.batch,self.config.gSz*self.config.gSz))

    self.tail = self.tail:cuda()
end

-- extra layers to relevance output
function DecompNet:build_extra()
    self.gradFit = nn.Sequential()
    self.gradFit:add(nn.Identity())
    self.gradFit = self.gradFit:cuda()
end

function DecompNet:test_forward(input_batch,layer)

end

function DecompNet:forward(input_batch)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    local pre = self.M:forward(self.inputs)
    local N = #self.M.modules
    local D = N - self.layer
    pre = pre * 100
    self.M:zeroGradParameters()
    for i=1,D do
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
