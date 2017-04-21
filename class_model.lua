--[[
    Forward and backward combined.
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

local DecompNet,_ = torch.class("nn.DecompNet",'nn.Container')

function DecompNet:__init(config,layer)
    print("Classification-Only Model from layer %d ." % layer)
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
    self.M:remove();self.M:remove();self.M:remove();
    -- if 4-remove, then start size is 1024,14,14
    -- 1024,8,8
    self.class = nn.Sequential()
    self.class:add(cudnn.SpatialAveragePooling(7, 7, 1, 1) ) -- same
    self.class:add(nn.View(2048))
    self.class:add(nn.Dropout(0.5)) -- there is no dropout in origin resnet
    self.class:add(nn.Linear(2048,90)) -- original resnet is 2048 -> 1000
    self.M:add(self.class)
    self.M = self.M:cuda()

    collectgarbage()

    -- pre-running to determine shapes
    self:precheck()
    self:build_tail()
end

function DecompNet:precheck()
    local tempin = torch.rand(1,3,224,224):cuda()
    self.M:training()
    local temp = self.M:forward(tempin)
    self.M:backward(tempin,temp)
    self.in_size = self.M.modules[1].gradInput:size(3)
    self.from_size = self.M.modules[self.layer].gradInput:size(3)
    self.from_nfeat = self.M.modules[self.layer].gradInput:size(2)
    self.scale = self.config.gSz / self.config.oSz 

    print("Pre-check :")
    print("Main Module output : ")
    print(temp:size())
    print("Relevance Origin : (%d,%d) " % {self.from_nfeat,self.from_size})
    print("Scale factor  %d " % self.scale)
end

function DecompNet:forward(input_batch)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    local pre = self.M:forward(self.inputs)

    self.output = pre
    return self.output
end

function DecompNet:backward(input_batch,gradInput)
    local gradIn
    cutorch.setDevice(self.gpu1)
    gradIn = self.M:backward(input_batch, gradInput)
    return gradIn
end

function DecompNet:relevance_forward(input_batch)
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

    wd = pre:clone()
    rele = torch.cmul(wd, self.M.modules[self.layer-1].output)

    return wd:sum(2), rele:sum(2)
end

function DecompNet:relevance_visualize(input)
    self.inputs = input
    local pred = self.M:forward(self.inputs):clone()
    local softmax = nn.SoftMax():cuda()
    pred = softmax:forward(pred)
    conf,ind = torch.max(pred)
    fwd = pred:clone():zeros()
    fwd[ind[1]]=1

    self.M:zeroGradParameters()
    self.full_decomp = self.M:backward(input_batch,pred*100)

    self.rele = torch.cmul(self.full_decomp,self.inputs)
    return self.rele:sum(2), self.full_decomp:sum(2)
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

function DecompNet:training()
    cutorch.setDevice(self.gpu1)
    self.M:training()
end

function DecompNet:evaluate()
    cutorch.setDevice(self.gpu1)
    self.M:evaluate()
end

function DecompNet:zeroGradParameters()
    cutorch.setDevice(self.gpu1)
    self.M:zeroGradParameters()
end

function DecompNet:updateParameters(lr,head)
    cutorch.setDevice(self.gpu1)
    self.M:updateParameters(lr)
end

function DecompNet:getParameters(head)
    -- Partial training
    if head == 3 then 
        print("Giving FT head parameters")
        return self.class:getParameters()
    end
    -- Full training
    print("Giving Model Full Parameters")
    return self.M:getParameters()
end

function DecompNet:cuda()
    cutorch.setDevice(self.gpu1)
    self.M:cuda()
end

function DecompNet:float()
    self.M:float()
end

function DecompNet:clone(...)
    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self); f:seek(1)
    local clone = f:readObject(); f:close()

    if select('#',...) > 0 then
        clone.M:share(self.M,...)
    end

    return clone
end

return nn.DecompNet