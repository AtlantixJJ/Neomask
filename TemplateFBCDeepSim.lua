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
    print("Decomp BaseNet from layer %d . Forward-Backward Combined. Naive." % layer)
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2
    self.layer = layer

    self.config = config
    self.name = config.name

    self.ks = 3
    self.pd = 1
    self.fs = 64

    cutorch.setDevice(self.gpu1)
    if #config.trans_model == 0 then
        -- the resnet model cannot be cuda model
        self.M = config.model
        self.M:remove();self.M:remove();self.M:remove();
        -- if 4-remove, then start size is 1024,14,14
        -- 1024,8,8
        self.M:add(cudnn.SpatialAveragePooling(7, 7, 1, 1) ) -- same
        self.M:add(nn.View(2048))
        self.M:add(nn.Dropout(0.5)) -- there is no dropout in origin resnet
        self.M:add(nn.Linear(2048,90)) -- original resnet is 2048 -> 1000
        self.M = self.M:cuda()
        collectgarbage()
    else
        print("Using Transfer Trained Model...")
        self.M = config.model:cuda()
        self.softmax = nn.SoftMax():cuda()
    end

    -- pre-running to determine shapes
    self:precheck()

    cutorch.setDevice(self.gpu2)
    self:build_extra() -- extra common layer
    self:build_scoreBranch()
    self:build_maskBranch()
    collectgarbage()

    self.maskNet = nn.Sequential():add(self.gradFit):add(self.maskBranch)
    self.scoreNet = nn.Sequential():add(self.gradFit):add(self.scoreBranch)
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

-- tail is used for clearing ups
function DecompNet:build_tail()
    -- scale up to input
    print("Scale %d : %d -> %d" % {self.scale, self.in_size, self.from_size})

    self.tail = nn.Sequential()

    self.tail:add(nn.SpatialUpSamplingBilinear(self.scale))
    -- need to convert to 1-d tensor to take in margin loss
    self.tail:add(nn.View(self.config.batch,self.config.oSz,self.config.oSz))

    self.tail = self.tail:cuda()
end

function DecompNet:build_maskBranch()
  print("Output raw size : %d" % self.from_size)
  local maskBranch = nn.Sequential()
  
  maskBranch:add(nn.Dropout(0.5))
  maskBranch:add(nn.Linear(512,self.config.oSz*self.config.oSz))
  maskBranch:add(nn.View(self.config.batch, self.config.oSz, self.config.oSz))

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

-- extra layers to relevance output
function DecompNet:build_extra()
    print("FingerPrint : blgx_fbc")
    self.gradFit = nn.Sequential()
    local forward_feat = nn.Sequential()
    local backward_feat = nn.Sequential()

    print("Top feature number : %d " % self.fs)

    forward_feat:add(cudnn.SpatialConvolution(self.from_nfeat,self.fs,1,1,1,1))
    forward_feat:add(nn.SpatialBatchNormalization(self.fs))
    forward_feat:add(cudnn.ReLU())

    backward_feat:add(cudnn.SpatialConvolution(self.from_nfeat,self.fs,1,1,1,1))
    backward_feat:add(nn.SpatialBatchNormalization(self.fs))
    backward_feat:add(cudnn.ReLU())

    self.gradFit:add(nn.ParallelTable():add(forward_feat):add(backward_feat))
    self.gradFit:add(nn.JoinTable(2)) -- join at (batch,x,256,256)

    self.gradFit:add(nn.SpatialZeroPadding(self.pd,self.pd))
    self.gradFit:add(cudnn.SpatialConvolution(2*self.fs,self.fs/2,self.ks,self.ks,1,1))
    self.gradFit:add(nn.SpatialBatchNormalization(self.fs/2))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit:add(nn.View(self.config.batch,self.fs/2*self.from_size*self.from_size))
    self.gradFit:add(nn.Dropout(.5))
    self.gradFit:add(nn.Linear(self.fs/2*self.from_size*self.from_size,512))
    self.gradFit:add(cudnn.ReLU())

    self.gradFit = self.gradFit:cuda()
end


function DecompNet:forward(input_batch,head)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    local pre = self.M:forward(self.inputs):clone()

    if head == 3 then
        self.output = pre
        return self.output
    end

    local N = #self.M.modules
    local D = N - self.layer + 1
    pre = self.softmax:forward(pre) * 10
    self.M:zeroGradParameters()

    for i=1,D do
        if N == i then
            pre = self.M.modules[N-i+1]:backward(self.inputs, pre)
        else
            pre = self.M.modules[N-i+1]:backward(self.M.modules[N-i].output, pre)
        end
    end

    --- build refine input
    cutorch.setDevice(self.gpu2)
    ------- this is the difference from Backward Only models
    self.ss_in = {self.M.modules[self.layer-1].output,self.M.modules[self.layer].gradInput}
    --------

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
    if head == 3 then
        cutorch.setDevice(self.gpu1)
        self.M:updateParameters(lr)
        print("Upd 3")
        return 3
    end
    
    cutorch.setDevice(self.gpu2)
    self.gradFit:updateParameters(lr)
    if head == 1 then
        self.maskBranch:updateParameters(lr)
        return 1
    elseif head == 2 then
        self.scoreBranch:updateParameters(lr)
        return 2
    end
end

function DecompNet:getParameters(head)
    if head == 3 then
        print("Giving Trunk Parameters")
        return self.M:getParameters()
    elseif head == 1 then
        print("Giving MaskBranch Parameters")
        return self.maskNet:getParameters()
    elseif head == 2 then
        print("Giving ScoreBranch Parameters")
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