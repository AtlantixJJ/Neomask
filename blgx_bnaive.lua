--[[
    This model is used to figure out which layer is more suitable for masking
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

local DecompNet,_ = torch.class("nn.DecompNet",'nn.Container')

function DecompNet:__init(config,layer)
    print("Decomp BaseNet from layer %d . Backward only and is naive" % layer)
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2
    self.layer = layer

    self.config = config
    self.name = config.name

    self.ks = 3
    self.pd = 1
    self.fs = 64

    cutorch.setDevice(self.gpu1)
    -- the preset model cannot be cuda model
    self.M = config.model
    -- for resnet with softmax layer, it will remove until
    self.M:remove();self.M:remove();self.M:remove();self.M:remove()
    -- 1024,8,8
    self.M:add(cudnn.SpatialAveragePooling(7, 7, 1, 1) )
    self.M:add(nn.View(2048))
    self.M:add(nn.Dropout(0.5)) -- there is no dropout in origin resnet
    self.M:add(nn.Linear(2048,80)) -- original resnet is 2048 -> 1000
    self.M:add(nn.SoftMax())
    self.M = self.M:cuda()
    collectgarbage()

    -- pre-running to determine shapes
    self:precheck()

    cutorch.setDevice(self.gpu2)
    self:build_extra()
    self:build_scoreBranch()
    self:build_maskBranch()

    self.maskNet = nn.Sequential():add(self.gradFit):add(self.maskBranch)
    self.scoreNet = nn.Sequential():add(self.gradFit):add(self.scoreBranch)
end

function DecompNet:build_maskBranch()
  print("Output raw size : %d" % self.from_size)
  local maskBranch = nn.Sequential()
  maskBranch:add(nn.Linear(512,56*56))
  maskBranch:add(nn.View(self.config.batch, 56, 56))
  self:build_tail()
  self.maskBranch = nn.Sequential():add(maskBranch:cuda()):add(self.tail)
end

function DecompNet:build_scoreBranch()
  local scoreBranch = nn.Sequential()
  scoreBranch:add(nn.Dropout(.5))
  scoreBranch:add(nn.Linear(512,1024))
  scoreBranch:add(nn.Threshold(0, 1e-6))

  scoreBranch:add(nn.Dropout(.5))
  scoreBranch:add(nn.Linear(1024,1))

  self.scoreBranch = scoreBranch:cuda()
  return self.scoreBranch
end

function DecompNet:precheck()
    local tempin = torch.rand(1,3,224,224):cuda()
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
    print("Scale %d : %d -> %d" % {4, self.in_size, self.from_size})

    self.tail = nn.Sequential()

    self.tail:add(nn.SpatialUpSamplingBilinear(4))
    -- need to convert to 1-d tensor to take in margin loss
    self.tail:add(nn.View(self.config.batch,56,56))

    self.tail = self.tail:cuda()
end

-- extra layers to relevance output
function DecompNet:build_extra()
    self.gradFit = nn.Sequential()

    self.gradFit:add(cudnn.SpatialConvolution(self.from_nfeat,64,1,1,1,1))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(64,16,self.ks,self.ks,1,1))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.View(self.config.batch,16*self.from_size*self.from_size))
    self.gradFit:add(nn.Dropout(.5))
    self.gradFit:add(nn.Linear(16*self.from_size*self.from_size,512))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit = self.gradFit:cuda()
end

function DecompNet:test_forward(input_batch,layer)

end

function DecompNet:forward(input_batch,head)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    local pre = self.M:forward(self.inputs)

    if head == 3 then
        self.output = pre
        return self.output
    end

    local N = #self.M.modules
    local D = N - self.layer + 1
    pre = pre * 100
    self.M:zeroGradParameters()

    for i=1,D do
        if N == i then
            pre = self.M.modules[N-i+1]:backward(self.inputs, pre)
        else
            pre = self.M.modules[N-i+1]:backward(self.M.modules[N-i].output, pre)
        end
    end

    -- Build refine input
    cutorch.setDevice(self.gpu2)
    self.ss_in = self.M.modules[self.layer].gradInput:clone()

    self.common_in = self.gradFit:forward(self.ss_in) -- :clone()

    if head == 1 then
        self.output = self.maskBranch:forward(self.common_in) -- :clone()
    elseif head == 2 then
        self.output = self.scoreBranch:forward(self.common_in)
    end
    -- print(self.output)
    return self.output
end

function DecompNet:backward(input_batch,gradInput,head)
    local gradIn
    if head == 3 then
        cutorch.setDevice(self.gpu1)
        gradIn = self.M:backward(input_batch, gradInput)
        return gradIn
    end

    cutorch.setDevice(self.gpu2)
    
    if head == 1 then
        gradIn = self.maskBranch:backward(self.common_in, gradInput)
    elseif head == 2 then
        gradIn = self.scoreBranch:backward(self.common_in, gradInput)
    end
    self.gradFitin = self.gradFit:backward(self.ss_in, gradIn) -- :clone()
    -- fdself.gradResin = self.M:backward(input_batch,self.gradFitin)
end

function DecompNet:training()
    cutorch.setDevice(self.gpu1)
    self.M:training()
    cutorch.setDevice(self.gpu2)
    self.gradFit:training()
    self.maskBranch:training()
    self.scoreBranch:training()
end

function DecompNet:evaluate()
    cutorch.setDevice(self.gpu1)
    self.M:evaluate()
    cutorch.setDevice(self.gpu2)
    self.gradFit:evaluate()
    self.maskBranch:evaluate()
    self.scoreBranch:evaluate()
end

function DecompNet:zeroGradParameters()
    cutorch.setDevice(self.gpu1)
    self.M:zeroGradParameters()
    cutorch.setDevice(self.gpu2)
    self.gradFit:zeroGradParameters()
    self.maskBranch:zeroGradParameters()
    self.scoreBranch:zeroGradParameters()
end

function DecompNet:updateParameters(lr,head)
    -- self.M:updateParameters(lr)
    cutorch.setDevice(self.gpu2)
    self.gradFit:updateParameters(lr)
    self.maskBranch:updateParameters(lr)
    self.scoreBranch:updateParameters(lr)
end

function DecompNet:getParameters(head)
    if head == 3 then
        return self.M:getParameters()
    elseif head == 1 then
        return self.maskNet:getParameters()
    elseif head == 2 then
        return self.scoreNet:getParameters()
    end
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
