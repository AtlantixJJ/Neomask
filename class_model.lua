--[[
    Forward and backward combined.
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

local ClassNet,_ = torch.class("nn.ClassNet",'nn.Container')

function ClassNet:__init(config,layer)
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
    self.M:add(cudnn.SpatialAveragePooling(7, 7, 1, 1) ) -- same
    self.M:add(nn.View(2048))
    self.M=self.M:cuda()
    self.scoreBranch = nn.Linear(2048,1):cuda()
    self.classBranch = nn.Linear(2048,90):cuda()
    self.classNet = nn.Sequential():add(self.M):add(self.classBranch)
    self.scoreNet = nn.Sequential():add(self.M):add(self.scoreBranch)

    collectgarbage()

    -- pre-running to determine shapes
    self:precheck()
    self:build_tail()
end

function ClassNet:precheck()
    local tempin = torch.rand(1,3,224,224):cuda()
    self.classNet:training()
    local temp = self.classNet:forward(tempin)
    self.classNet:backward(tempin,temp)
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

function ClassNet:forward(input_batch,head)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    local pre = self.M:forward(self.inputs)
    if head == 2 then --score
        self.output = self.scoreBranch:forward(pre)
    elseif head == 3 then --class
        self.output = self.classBranch:forward(pre)
    end

    return self.output
end

function ClassNet:backward(input_batch,gradInput,head)
    local gradIn
    cutorch.setDevice(self.gpu1)
    if head == 2 then
        gradIn = self.scoreNet:backward(input_batch, gradInput)
    elseif head == 3 then
        gradIn = self.classNet:backward(input_batch, gradInput)
    end
    return gradIn
end

function ClassNet:relevance_forward(input_batch)
    self.inputs = input_batch
    local pre = self.classNet:forward(self.inputs)

    if head == 3 then
        self.output = pre
        return self.output
    end

    local N = #self.classNet.modules
    local D = N - self.layer + 1
    pre = pre * 100

    self.M:zeroGradParameters()
    for i=1,D do
        if N == i then
            pre = self.classNet.modules[N-i+1]:backward(self.inputs, pre)
        else
            pre = self.classNet.modules[N-i+1]:backward(self.classNet.modules[N-i].output, pre)
        end
    end

    wd = pre:clone()
    rele = torch.cmul(wd, self.classNet.modules[self.layer-1].output)

    return wd:sum(2), rele:sum(2)
end

function ClassNet:relevance_visualize(input)
    self.inputs = input
    local pred = self.classNet:forward(self.inputs):clone()
    local softmax = nn.SoftMax():cuda()
    pred = softmax:forward(pred)
    conf,ind = torch.max(pred)
    fwd = pred:clone():zeros()
    fwd[ind[1]]=1

    self.classNet:zeroGradParameters()
    self.full_decomp = self.classNet:backward(input_batch,pred*100)

    self.rele = torch.cmul(self.full_decomp,self.inputs)
    return self.rele:sum(2), self.full_decomp:sum(2)
end

-- tail is used for clearing ups
function ClassNet:build_tail()
    -- scale up to input
    print("Scale %d : %d -> %d" % {4, self.in_size, self.from_size})

    self.tail = nn.Sequential()

    self.tail:add(nn.SpatialUpSamplingBilinear(4))
    -- need to convert to 1-d tensor to take in margin loss
    self.tail:add(nn.View(self.config.batch,56,56))

    self.tail = self.tail:cuda()
end

function ClassNet:training()
    cutorch.setDevice(self.gpu1)
    self.classNet:training()
    self.scoreNet:training()
end

function ClassNet:evaluate()
    cutorch.setDevice(self.gpu1)
    self.classNet:evaluate()
    self.scoreNet:evaluate()
end

function ClassNet:zeroGradParameters()
    cutorch.setDevice(self.gpu1)
    self.classNet:zeroGradParameters()
    self.scoreNet:zeroGradParameters()
end

function ClassNet:updateParameters(lr,head)
    cutorch.setDevice(self.gpu1)
    self.classNet:updateParameters(lr)
    self.scoreNet:updateParameters(lr)
end

function ClassNet:getParameters(head)
    -- Partial training
    if head == 3 then --head parameters
        print("Giving class head parameters")
        return self.classBranch:getParameters()
    elseif head == 4 then
        -- Full training
        print("Giving Full Classification Parameters")
        return self.classNet:getParameters()
    elseif head == 5 then
        print("Giving Full Score Parameters")
        return self.scoreNet:getParameters()
    elseif head == 2 then
        print("Giving Score head parameters")
        return self.scoreBranch:getParameters()
    end
end

function ClassNet:cuda()
    cutorch.setDevice(self.gpu1)
    self.scoreNet:cuda()
    self.classNet:cuda()
end

function ClassNet:float()
    self.M:float()
end

function ClassNet:clone(...)
    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self); f:seek(1)
    local clone = f:readObject(); f:close()

    if select('#',...) > 0 then
        clone.M:share(self.M,...)
    end

    return clone
end

return nn.ClassNet