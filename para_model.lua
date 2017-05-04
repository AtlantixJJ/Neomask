--[[
    Forward and backward combined.
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'
paths.dofile('SpatialSymmetricPadding.lua')
local ParallelNet,_ = torch.class("nn.ParallelNet",'nn.Container')

function ParallelNet:__init(config,traceback_depth)
    print("Parallel Model on traceback depth %d ." % traceback_depth)
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2
    self.traceback_depth = traceback_depth - 1

    self.config = config
    self.name = config.name

    self.ks = 3
    self.pd = 1
    self.fs = 64

    cutorch.setDevice(self.gpu1)

    self.softmax = nn.SoftMax():cuda()
    -- the preset model cannot be cuda model
    self.M = config.model
    self.M:remove();self.M:remove();self.M:remove();
    -- if 4-remove, then start size is 1024,14,14
    -- 1024,8,8
    self.trunk_head = nn.Sequential()
    for i=1,self.traceback_depth do
        self.trunk_head:add(self.M.modules[#self.M])
        self.M:remove()
    end
    self.trunk = nn.Sequential():add(self.M):cuda()
    self.trunk_head:add(cudnn.SpatialAveragePooling(7, 7, 1, 1) ) -- same
    self.trunk_head:add(nn.View(2048))
    self.trunk_head = self.trunk_head:cuda()
    
    self.trunkNet = nn.Sequential():add(self.trunk):add(self.trunk_head)
    collectgarbage()
    
    self.scoreBranch = nn.Sequential():add(nn.Linear(2048,1)):cuda()
    self.classBranch = nn.Sequential():add(nn.Linear(2048,90)):cuda()
    self.trunk_head_class = nn.Sequential():add(self.trunk_head):add(self.classBranch)
    -- pre-running to determine shapes
    self:precheck()
    self:build_tail()
    self:build_maskBranch()

    self.classNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.classBranch)
    self.scoreNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.scoreBranch)
    self.maskNet = nn.Sequential():add(self.trunk ):add(self.trunk_head):add(self.maskBranch)
    collectgarbage()

    self:make_NewActivation() 
end

-- tail is used for clearing ups
function ParallelNet:build_tail()
    -- scale up to input
    print("Scale 4")
    self.tail = nn.Sequential()

    self.tail:add(nn.SpatialUpSamplingBilinear(4))
    -- need to convert to 1-d tensor to take in margin loss
    self.tail:add(nn.View(self.config.batch,56,56))

    self.tail = self.tail:cuda()
end

-- take relevance and original input as input, output the activation result
-- simulate LSTM forget gate
function ParallelNet:make_NewActivation()
    local input_layer = nn.ParallelTable()
    input_layer:add(nn.Identity()) -- nn.CMul(self.from_nfeat, self.from_size, self.from_size) -- use multiple
    input_layer:add(nn.Sequential():add(nn.Mul()):add(cudnn.Tanh)) -- use single scale factor
    self.new_act = nn.Sequential():add(nn.CMulTable(input_layer)):cuda()
end

function ParallelNet:build_maskBranch()
    local maskBranch = nn.Sequential()
    maskBranch:add(nn.Dropout(.5))
    maskBranch:add(nn.Linear(2048,512))
    maskBranch:add(nn.Dropout(.5))
    maskBranch:add(nn.Linear(512,56*56))
    maskBranch:add(nn.View(self.config.batch, 56, 56))
    self.maskBranch = nn.Sequential():add(maskBranch:cuda()):add(self.tail)
end

function ParallelNet:precheck()
    local tempin = torch.rand(1,3,224,224):cuda()
    local trunk_out = self.trunk:forward( tempin )
    local pred = self.softmax:forward(self.trunk_head_class:forward(trunk_out))
    self.trunk_head_class:training()
    local rele_input = self.trunk_head_class:c_backward(trunk_out, pred)

    self.in_size = 224
    self.from_size = rele_input:size(3)
    self.from_nfeat = rele_input:size(2)
    self.scale = self.config.gSz / self.config.oSz 

    print("Pre-check :")
    print("Class Module output : ")
    print(pred:size())
    print("Relevance Origin : (%d, %d, %d) " % {self.from_nfeat, self.from_size, self.from_size})
    print("Scale factor  %d " % self.scale)
end

-- need to set self.ss_in
function ParallelNet:forward_mask()
    self.rele_output =  self.new_act:forward( self.ss_in )
    self.mod_trunkout = self.trunk_head:forward( self.rele_output )
    return self.maskBranch:forward(self.mod_trunkout)
end

function ParallelNet:forward(input_batch,head,iftrain)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch
    if head == 1 then --mask
        local trunk_out = self.trunk:forward( self.inputs )
        -- trunk_head is classification head
        if iftrain == true then
            self.trunk_head_class:training()
        else self.trunk_head_class:evaluate() end

        local pred = self.softmax:forward(self.trunk_head_class:forward(trunk_out))
        self.trunk_head_class:training()
        local rele_input = self.trunk_head_class:c_backward(trunk_out, pred)
        -- modified trunk head output
        -- use relevance as input
        self.ss_in = {trunk_out, rele_input}
        self.output = self:forward_mask()
    elseif head == 2 then --score
        self.branch_input = self.trunkNet:forward(self.inputs)
        self.output = self.scoreBranch:forward(self.branch_input)
    elseif head == 3 then --class
        self.branch_input = self.trunkNet:forward(self.inputs)
        self.output = self.classBranch:forward(self.branch_input)
    end

    return self.output
end

function ParallelNet:backward(input_batch,gradInput,head,ifBranch)
    local gradIn
    cutorch.setDevice(self.gpu1)
    if head == 1 then
        gradInput = self.maskBranch:backward(self.mod_trunkout, gradInput)
        gradInput = self.trunk_head:backward(self.rele_output, gradInput)
        gradInput = self.new_act:backward(self.ss_in, gradInput)
        gradIn = self.trunk:backward(input_batch, self.ss_in[1])
        -- the other forward_prop is ignored
    elseif head == 2 then
        if ifBranch then
            gradIn = self.scoreBranch:backward(self.branch_input, gradInput)
        else
            gradIn = self.scoreNet:backward(input_batch, gradInput)
        end
    elseif head == 3 then
        if ifBranch then
            gradIn = self.classBranch:backward(self.branch_input, gradInput)
        else
            gradIn = self.classNet:backward(input_batch, gradInput)
        end
    end
    return gradIn
end

function ParallelNet:relevance_forward(input_batch)
    self.inputs = input_batch
    local pre = self.ParallelNet:forward(self.inputs)

    if head == 3 then
        self.output = pre
        return self.output
    end

    local N = #self.ParallelNet.modules
    local D = N - self.layer + 1
    pre = pre * 100

    self.M:zeroGradParameters()
    for i=1,D do
        if N == i then
            pre = self.ParallelNet.modules[N-i+1]:backward(self.inputs, pre)
        else
            pre = self.ParallelNet.modules[N-i+1]:backward(self.ParallelNet.modules[N-i].output, pre)
        end
    end

    wd = pre:clone()
    rele = torch.cmul(wd, self.ParallelNet.modules[self.layer-1].output)

    return wd:sum(2), rele:sum(2)
end

function ParallelNet:relevance_visualize(input)
    self.inputs = input
    local pred = self.ParallelNet:forward(self.inputs):clone()
    local softmax = nn.SoftMax():cuda()
    pred = softmax:forward(pred)
    conf,ind = torch.max(pred)
    fwd = pred:clone():zeros()
    fwd[ind[1]]=1

    self.ParallelNet:zeroGradParameters()
    self.full_decomp = self.ParallelNet:backward(input_batch,pred*100)

    self.rele = torch.cmul(self.full_decomp,self.inputs)
    return self.rele:sum(2), self.full_decomp:sum(2)
end

function ParallelNet:training()
    cutorch.setDevice(self.gpu1)
    self.trunk:training()
    self.trunk_head:training()
    self.new_act:training()
    self.maskBranch:training()
    self.classBranch:training()
    self.scoreBranch:training()
end

function ParallelNet:evaluate()
    cutorch.setDevice(self.gpu1)
    self.trunk:evaluate()
    self.trunk_head:evaluate()
    self.new_act:evaluate()
    self.classBranch:evaluate()
    self.maskBranch:evaluate()
    self.scoreBranch:evaluate()
end

function ParallelNet:zeroGradParameters()
    cutorch.setDevice(self.gpu1)
    self.trunkNet:zeroGradParameters()
    self.new_act:zeroGradParameters()
    self.classBranch:zeroGradParameters()
    self.maskBranch:zeroGradParameters()
    self.scoreBranch:zeroGradParameters()
end

function ParallelNet:updateParameters(lr)
    cutorch.setDevice(self.gpu1)
    self.trunk:updateParameters(lr)
    self.trunk_head:updateParameters(lr)
    self.new_act:updateParameters(lr)
    self.classBranch:updateParameters(lr)
    self.maskBranch:updateParameters(lr)
    self.scoreBranch:updateParameters(lr)
end

function ParallelNet:cuda()
    cutorch.setDevice(self.gpu1)
    self.trunk:cuda()
    self.trunk_head:cuda()
    self.new_act:cuda()
    self.classBranch:cuda()
    self.maskBranch:cuda()
    self.scoreBranch:cuda()
end

function ParallelNet:float()
    self.trunk:float()
    self.trunk_head:float()
    self.new_act:float()
    self.classBranch:float()
    self.maskBranch:float()
    self.scoreBranch:float()
end

function ParallelNet:clone(...)
    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self); f:seek(1)
    local clone = f:readObject(); f:close()

    if select('#',...) > 0 then
        clone.trunk:share(self.trunk,...)
        clone.trunk_head:share(self.trunk_head,...)
        clone.new_act:share(self.new_act,...)
        clone.classBranch:share(self.classBranch,...)
        clone.scoreBranch:share(self.scoreBranch,...)
        clone.maskBranch:share(self.maskBranch,...)
    end

    return clone
end

return nn.ParallelNet